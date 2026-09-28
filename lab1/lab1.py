"""Лабораторная №1: анализ Twitter US Airline Sentiment."""

import argparse
import csv
import html
import math
import random
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
NUMBER = re.compile(r"\b\d+(?:[.,:]\d+)*\b")
HTML_ENTITY = re.compile(r"&(?:\#\d+|\#x[0-9a-f]+|[a-z]+);", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
TOKEN = re.compile(r"\[[A-Z]+\]|[a-z]+(?:['’][a-z]+)?|[.!?]+|[,;:]", re.I)
SPECIAL = {"<s>", "</s>", "[UNK]", "[URL]", "[USER]", "[TAG]", "[NUM]", "[EMOJI]", "[EMAIL]"}


def clean(text):
    text = html.unescape(text)
    text = URL.sub(" [URL] ", text)
    text = EMAIL.sub(" [EMAIL] ", text)
    text = MENTION.sub(" [USER] ", text)
    text = HASHTAG.sub(" [TAG] ", text)
    text = EMOJI.sub(" [EMOJI] ", text)
    text = NUMBER.sub(" [NUM] ", text)
    tokens = TOKEN.findall(text.lower())
    return ["<s>", *[token.upper() if token.startswith("[") else token for token in tokens], "</s>"]


def levenshtein(a, b):
    """Две строки ДП: O(min(len(a), len(b))) памяти."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def typo_pairs(frequencies, limit=10):
    """Индекс по удалению 1–2 букв сокращает поиск кандидатов."""
    words = [w for w, n in frequencies.items() if w.isalpha() and 4 <= len(w) <= 18]
    index = defaultdict(set)
    for word in words:
        keys = {word}
        for i in range(len(word)):
            one = word[:i] + word[i + 1:]
            keys.add(one)
            for j in range(len(one)):
                keys.add(one[:j] + one[j + 1:])
        for key in keys:
            index[key].add(word)
    pairs = set()
    for group in index.values():
        rare = [w for w in group if frequencies[w] <= 2]
        common = [w for w in group if frequencies[w] >= 5]
        for a in rare:
            for b in common:
                if a != b and abs(len(a) - len(b)) <= 2:
                    pairs.add((a, b))
    result = [(levenshtein(a, b), a, b, frequencies[a], frequencies[b]) for a, b in pairs]
    return sorted((row for row in result if row[0] in (1, 2)),
                  key=lambda row: (row[0], -row[4], row[3], row[1]))[:limit]


class BigramModel:
    def __init__(self, documents, min_count=2):
        counts = Counter(t for doc in documents for t in doc[1:-1])
        self.vocabulary = {t for t, count in counts.items() if count >= min_count}
        self.vocabulary.update({"</s>", "[UNK]"})
        self.transitions = defaultdict(Counter)
        self.contexts = Counter()
        for document in documents:
            mapped = self.map_document(document)
            for a, b in zip(mapped, mapped[1:]):
                self.transitions[a][b] += 1
                self.contexts[a] += 1

    def map_document(self, document):
        return ["<s>", *(t if t in self.vocabulary else "[UNK]" for t in document[1:-1]), "</s>"]

    def perplexity(self, documents, alpha=0):
        total_log, n = 0.0, 0
        for document in documents:
            mapped = self.map_document(document)
            for a, b in zip(mapped, mapped[1:]):
                count = self.transitions[a][b]
                context = self.contexts[a]
                if alpha == 0 and count == 0:
                    return math.inf
                probability = ((count + alpha) / (context + alpha * len(self.vocabulary))
                               if alpha else count / context)
                total_log += math.log(probability)
                n += 1
        return math.exp(-total_log / n)

    def generate(self, rng, max_tokens=30):
        current, tokens = "<s>", ["<s>"]
        for _ in range(max_tokens):
            options = self.transitions[current]
            if not options:
                break
            current = rng.choices(list(options), weights=list(options.values()))[0]
            tokens.append(current)
            if current == "</s>":
                break
        return " ".join(tokens)


def prepare_nltk():
    target = Path.home() / "nltk_data"
    for category, name in (("corpora", "wordnet"),
                           ("taggers", "averaged_perceptron_tagger_eng")):
        archive = target / category / f"{name}.zip"
        extracted = target / category / name
        if not archive.exists() and not extracted.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            url = ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
                   f"packages/{category}/{name}.zip")
            urllib.request.urlretrieve(url, archive)
        if category == "taggers" and not extracted.exists():
            with zipfile.ZipFile(archive) as source:
                source.extractall(archive.parent)
    wordnet.ensure_loaded()


def normalization(tokens):
    stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
    words = sorted({t for doc in tokens for t in doc if t.isalpha()})
    tagged = nltk.pos_tag(words)
    pos_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    stems = {stemmer.stem(w) for w in words}
    lemmas = {lemmatizer.lemmatize(w, pos_map.get(pos[0], wordnet.NOUN)) for w, pos in tagged}
    return len(words), len(stems), len(lemmas)


def save_charts(texts, output):
    lengths = [len(re.findall(r"\b[a-z]+(?:['’][a-z]+)?\b", t.lower())) for t in texts]
    chars = [len(t) for t in texts]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lengths, bins=30)
    axes[0].set(xlabel="Слов в твите", ylabel="Число твитов")
    axes[1].hist(chars, bins=30)
    axes[1].set(xlabel="Символов в твите", ylabel="Число твитов")
    fig.tight_layout()
    fig.savefig(output / "lengths.png", dpi=140)
    plt.close(fig)
    stops = Counter(w for t in texts for w in re.findall(r"[a-z]+", t.lower())
                    if w in {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "is", "are", "was", "were", "be", "been", "i", "you", "we", "they", "it", "my", "your", "our", "this", "that", "have", "has", "had", "do", "does", "did", "not", "no", "so", "as", "from", "by", "me", "us", "he", "she", "am", "will", "can", "if", "just", "all", "up", "what", "when", "how", "why", "who"})
    top = stops.most_common(30)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([w for w, _ in top], [n for _, n in top])
    ax.tick_params(axis="x", rotation=65)
    ax.set(ylabel="Вхождений", title="30 наиболее частых стоп-слов (Train)")
    fig.tight_layout()
    fig.savefig(output / "stopwords.png", dpi=140)
    plt.close(fig)
    return sum(lengths) / len(lengths), sum(chars) / len(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Tweets.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, args.data)
    frame = pd.read_csv(args.data, usecols=["text", "airline_sentiment"]).dropna()
    train, test = train_test_split(frame, test_size=0.2, random_state=42,
                                   stratify=frame["airline_sentiment"])
    train_texts, test_texts = train.text.tolist(), test.text.tolist()
    mean_words, mean_chars = save_charts(train_texts, args.output)
    noise = {"URL": URL, "упоминания": MENTION, "хештеги": HASHTAG,
             "HTML-сущности": HTML_ENTITY, "эмодзи": EMOJI, "числа": NUMBER}
    noise_counts = {name: sum(bool(pattern.search(t)) for t in train_texts)
                    for name, pattern in noise.items()}
    train_tokens = [clean(t) for t in train_texts]
    test_tokens = [clean(t) for t in test_texts]
    prepare_nltk()
    before, after_stem, after_lemma = normalization(train_tokens)
    frequencies = Counter(t for doc in train_tokens for t in doc[1:-1])
    pairs = typo_pairs(frequencies)
    model = BigramModel(train_tokens)
    plain = model.perplexity(test_tokens)
    smoothed = model.perplexity(test_tokens, alpha=1)
    examples = [model.generate(random.Random(42 + i)) for i in range(5)]
    lines = ["# Отчёт: Twitter US Airline Sentiment", "",
             "## Выборка и структура", "",
             f"После удаления пустых значений: {len(frame)}. Train: {len(train)}, Test: {len(test)}; "
             "разбиение 80/20, стратификация по `airline_sentiment`, seed=42.", "",
             "| Класс | Train | Test |", "|---|---:|---:|"]
    for label in sorted(frame.airline_sentiment.unique()):
        lines.append(f"| {label} | {sum(train.airline_sentiment == label)} | {sum(test.airline_sentiment == label)} |")
    lines += ["", f"Средняя длина Train: {mean_words:.2f} слов и {mean_chars:.2f} символов; "
              "длины измерены до очистки.", "", "![Длины](lengths.png)", "",
              "![Стоп-слова](stopwords.png)", "", "## Шум и очистка", "",
              "Число документов Train с соответствующим видом шума (виды пересекаются):", "",
              "| Вид | Документов |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in noise_counts.items()]
    lines += ["", "URL, почта, упоминания, хештеги, числа и эмодзи заменяются отдельными "
              "токенами. HTML-сущности декодируются; пунктуация `.!?,;:` сохраняется. "
              "Остальные символы отсекаются при токенизации. Границы `<s>` и `</s>` "
              "добавляются после очистки. Хештеги заменяются целиком: это уменьшает "
              "словарь, но теряет смысл их текста.", "", "## Нормализация", "",
              "Сравнение словарей только по буквенным токенам Train, без пунктуации и "
              "специальных меток. Лемматизация WordNet учитывает часть речи, "
              "оценённую NLTK; ошибки разметки влияют на результат.", "",
              "| Исходный | После стемминга Porter | После лемматизации WordNet |",
              "|---:|---:|---:|", f"| {before} | {after_stem} | {after_lemma} |", "",
              "Стемминг сильнее объединяет формы, но может получать несловарные основы; "
              "лемматизация удобнее для чтения. Для биграммной модели ниже остаются "
              "очищенные исходные формы, чтобы не разрушать поверхностную грамматику.", "",
              "## Пары похожих слов", "",
              "Кандидаты: редкое слово (частота ≤2) и частое (≥5), расстояние 1 или 2. "
              "Это кандидаты на опечатки, а не доказанные ошибки; среди них бывают разные слова.", "",
              "| Расстояние | Редкое слово | Частое слово | Частоты |", "|---:|---|---|---:|"]
    lines += [f"| {d} | {a} | {b} | {na} / {nb} |" for d, a, b, na, nb in pairs]
    lines += ["", "## Биграммная модель", "",
              f"Размер словаря предсказываемых токенов: {len(model.vocabulary)}. "
              "Слова с частотой <2 в Train заменены на `[UNK]` при обучении; "
              "невиданные слова Test тоже отображаются в `[UNK]`. `<s>` служит "
              "только контекстом, `</s>` предсказывается.", "",
              "Без сглаживания: P(w|c)=count(c,w)/count(c); "
              "при неизвестном переходе вероятность нулевая. "
              "Со сглаживанием Лапласа: P(w|c)=(count(c,w)+1)/(count(c)+|V|). "
              "Перплексия: exp(-Σ log P(wᵢ|wᵢ₋₁)/M), M включает конечный токен "
              "каждого тестового документа.", "",
              "| Режим | Перплексия Test |", "|---|---:|",
              f"| Без сглаживания | {'∞' if math.isinf(plain) else f'{plain:.2f}'} |",
              f"| Лаплас (+1) | {smoothed:.2f} |", "",
              "Если встретился хотя бы один отсутствующий в Train переход, "
              "правдоподобие всей тестовой выборки без сглаживания равно нулю, "
              "поэтому перплексия бесконечна. Сглаживание назначает положительную "
              "вероятность каждому токену словаря; численное сравнение с ∞ "
              "не означает, что модель стала качественной по смыслу.", "",
              "## Генерация (seed=42…46)", ""]
    lines += [f"{i}. `{example}`" for i, example in enumerate(examples, 1)]
    lines += ["", "Биграммы воспроизводят соседние сочетания, но не удерживают "
              "синтаксис и тему на длине всего твита; `[UNK]` и заменители шума "
              "снижают читаемость. Остановка происходит на `</s>` или после 30 "
              "сгенерированных токенов.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Train={len(train)} Test={len(test)} V={len(model.vocabulary)} "
          f"PPL(no smoothing)={plain} PPL(Laplace)={smoothed:.2f}")
    print(f"Отчёт и графики: {args.output}")


if __name__ == "__main__":
    main()
