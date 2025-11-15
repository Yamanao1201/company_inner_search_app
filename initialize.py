"""
このファイル（initialize.py）は、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import unicodedata

import pandas as pd

from dotenv import load_dotenv
import streamlit as st

from langchain.schema import Document as LCDocument
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from sudachipy import dictionary, tokenizer as sudachi_tokenizer

import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# Sudachi（日本語形態素解析）の準備
############################################################
_sudachi_tokenizer = dictionary.Dictionary().create()
_SUDACHI_MODE = sudachi_tokenizer.Tokenizer.SplitMode.C  # C: やや長めの単位で分割


def sudachi_tokenize(text: str) -> list[str]:
    """
    日本語テキストを Sudachi で分かち書きし、トークンのリストを返す。
    BM25Retriever で使用する前処理関数。
    """
    if not isinstance(text, str):
        text = str(text)
    return [m.surface() for m in _sudachi_tokenizer.tokenize(text, _SUDACHI_MODE)]


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにハンドラが設定されている場合は二重設定を防ぐ
    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )

    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    log_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def initialize_retriever():
    """
    画面読み込み時にRAGのRetrieverを作成する。

    - PDF / DOCX / CSV / TXT などを読み込み
    - 社員名簿CSVは部署ごとに1ドキュメントにまとめて整形
    - CharacterTextSplitterでチャンク分割
    - ベクトル検索（Chroma） + BM25（Sudachi + rank_bm25）を
      EnsembleRetriever でハイブリッド化
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでに作成済みなら再作成しない
    if "retriever" in st.session_state:
        return

    # 1. データソース読み込み（PDF / DOCX / 社員名簿CSV など）
    docs_all = load_data_sources()

    # 2. Windows環境での文字化け対策などの文字列調整
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 3. ベクトル用の埋め込みモデル
    embeddings = OpenAIEmbeddings()

    # 4. チャンク分割（ベクトル検索 / BM25検索の両方でこの単位を使用）
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=ct.TEXT_SPLITTER_CHUNK_OVERLAP,
        separator="\n"
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # 5. ベクトルストア（Chroma）を作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # 6. ベクトル検索用 Retriever
    vector_retriever = db.as_retriever(
        search_kwargs={"k": ct.RETRIEVER_SEARCH_TOP_K}
    )

    # 7. BM25（キーワード）検索用 Retriever
    #    ※内部で rank_bm25 を利用。前処理に Sudachi を使って日本語対応。
    bm25_retriever = BM25Retriever.from_documents(
        splitted_docs,
        preprocess_func=sudachi_tokenize
    )
    bm25_retriever.k = ct.RETRIEVER_SEARCH_TOP_K

    # 8. ベクトル＋BM25 のハイブリッド Retriever（EnsembleRetriever）
    #    weights でベクトル検索とBM25検索のスコアの重みを調整
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[ct.VECTOR_SEARCH_WEIGHT, ct.BM25_SEARCH_WEIGHT],
    )

    # 9. アプリ全体から利用する Retriever として保存
    st.session_state.retriever = ensemble_retriever

    logger.info("Hybrid retriever (Vector + BM25 with Sudachi) has been initialized.")


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース（Documentのリスト）
    """
    docs_all: list[LCDocument] = []

    # ローカルフォルダ配下のファイル（PDF / DOCX / CSV / TXT など）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    # Webページ（constants.WEB_URL_LOAD_TARGETS で指定）
    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        web_docs_all.extend(web_docs)

    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path: str, docs_all: list[LCDocument]):
    """
    RAGの参照先となるデータソースの読み込み（再帰的にフォルダを探索）

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def load_employee_master_csv(path: str) -> list[LCDocument]:
    """
    社員名簿CSV（社員名簿.csv）専用のローダー。
    各行をそのままドキュメント化するのではなく、
    【部署ごとに1つのドキュメント】にまとめ、
    「女性職員」「正社員」「経済学部卒」などの検索キーワードを
    意識したテキストに整形する。

    Args:
        path: CSVファイルパス

    Returns:
        LangChainの Document オブジェクトのリスト
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.error(f"{ct.CSV_ERROR_MESSAGE}\n{e}")
        return []

    df = df.fillna("")

    docs: list[LCDocument] = []

    # 部署ごとにグルーピング
    for dept, group in df.groupby("部署"):
        lines: list[str] = []

        # ドキュメントのヘッダー部分（自然文）
        lines.append(f"部署: {dept}")
        lines.append(f"このドキュメントは{dept}に所属している従業員情報の一覧である。")
        lines.append(
            f"{dept}に所属している正社員・契約社員・派遣社員・アルバイトなどの従業員情報をまとめている。"
        )
        lines.append(
            "女性職員・男性職員の区別や、大学名・学部・学科（例: 経済学部卒）などの学歴情報も含まれている。"
        )
        lines.append("")  # 空行

        # 各従業員ごとの行を作成
        for _, row in group.iterrows():
            emp_id = row["社員ID"]
            name = row["氏名（フルネーム）"]
            gender = row["性別"]
            gender_label = "女性職員" if gender == "女性" else "男性職員"

            emp_type = row["従業員区分"]  # 正社員, 契約社員, 派遣, アルバイトなど
            emp_type_label = f"{emp_type}として在籍している従業員"

            position = row["役職"]
            mail = row["メールアドレス"]
            birth = row["生年月日"]
            age = row["年齢"]
            joined = row["入社日"]

            university = row.get("大学名", "")
            faculty = row.get("学部・学科", "")
            faculty_label = f"{faculty}卒" if faculty else ""

            skills = row.get("スキルセット", "")
            qualifications = row.get("保有資格", "")

            # 1人分の情報を自然文に近い形で整形
            line_parts: list[str] = [
                f"従業員ID: {emp_id}",
                f"氏名: {name}",
                f"性別: {gender}（{gender_label}）",
                f"従業員区分: {emp_type}（{emp_type_label}）",
                f"部署: {dept}",
                f"役職: {position}",
                f"生年月日: {birth}",
                f"年齢: {age}歳",
                f"入社日: {joined}",
            ]

            if university:
                line_parts.append(f"大学名: {university}")
            if faculty:
                # 学部・学科と「◯◯学部卒」の両方を書いておく
                line_parts.append(f"学部・学科: {faculty}")
                if faculty_label:
                    line_parts.append(f"学歴情報: {faculty_label}")
            if skills:
                line_parts.append(f"スキルセット: {skills}")
            if qualifications:
                line_parts.append(f"保有資格: {qualifications}")
            if mail:
                line_parts.append(f"メールアドレス: {mail}")

            # " / " で連結して1行のテキストに
            line = " / ".join(line_parts)

            # 箇条書き形式で追加
            lines.append(f"- {line}")

        page_content = "\n".join(lines)

        doc = LCDocument(
            page_content=page_content,
            metadata={
                "source": path,
                "部署": dept
            }
        )
        docs.append(doc)

    return docs


def file_load(path: str, docs_all: list[LCDocument]):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    file_extension = os.path.splitext(path)[1]
    file_name = os.path.basename(path)

    # ★ 社員名簿CSVだけ特別扱い（部署ごとドキュメント統合＋テキスト調整）
    if file_name == "社員名簿.csv":
        employee_docs = load_employee_master_csv(path)
        docs_all.extend(employee_docs)
        return

    # それ以外は従来どおり SUPPORTED_EXTENSIONS を使う
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整

    Args:
        s: 調整を行う文字列

    Returns:
        調整を行った文字列
    """
    if type(s) is not str:
        return s

    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s

    return s