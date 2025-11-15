"""
このファイル（components.py）は、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    回答モードのラジオボタンを表示
    """
    # ラジオボタンをサイドバーに表示
    with st.sidebar:
        st.session_state.mode = st.radio(
            label="利用目的を選択",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2]
        )


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    # AIメッセージの初期表示をサイドバーに表示
    with st.sidebar:
        st.markdown("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")

        st.divider()

        st.markdown("**【「社内文書検索」を選択した場合】**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.markdown(message["content"])

            else:
                if message["content"]["mode"] == ct.ANSWER_MODE_1:

                    if not "no_file_path_flg" in message["content"]:
                        # ==========================================
                        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
                        # ==========================================
                        st.markdown(message["content"]["main_message"])

                        icon = utils.get_source_icon(message['content']['main_file_path'])
                        if "main_page_number" in message["content"]:
                            st.success(f"{message['content']['main_file_path']} (ページNo.{message['content']['main_page_number']})", icon=icon)
                        else:
                            st.success(f"{message['content']['main_file_path']}", icon=icon)

                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"]:
                            st.markdown(message["content"]["sub_message"])

                            for sub_choice in message["content"]["sub_choices"]:
                                icon = utils.get_source_icon(sub_choice['source'])
                                if "page_number" in sub_choice:
                                    st.info(f"{sub_choice['source']} (ページNo.{sub_choice['page_number']})", icon=icon)
                                else:
                                    st.info(f"{sub_choice['source']}", icon=icon)
                    else:
                        st.markdown(message["content"]["answer"])

                else:
                    st.markdown(message["content"]["answer"])

                    if "file_info_list" in message["content"]:
                        st.divider()
                        st.markdown(f"##### {message['content']['message']}")
                        for file_info in message["content"]["file_info_list"]:
                            icon = utils.get_source_icon(file_info)
                            st.info(file_info, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:

        # ==========================================
        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
        # ==========================================
        main_file_path = llm_response["context"][0].metadata["source"]

        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)

        icon = utils.get_source_icon(main_file_path)
        if "page" in llm_response["context"][0].metadata:
            main_page_number = llm_response["context"][0].metadata["page"]
            st.success(f"{main_file_path} (ページNo.{main_page_number})", icon=icon)
        else:
            st.success(f"{main_file_path}", icon=icon)

        # ==========================================
        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
        # ==========================================
        sub_choices = []
        duplicate_check_list = []

        for document in llm_response["context"][1:]:
            sub_file_path = document.metadata["source"]

            if sub_file_path == main_file_path:
                continue

            if sub_file_path in duplicate_check_list:
                continue

            duplicate_check_list.append(sub_file_path)

            if "page" in document.metadata:
                sub_page_number = document.metadata["page"]
                sub_choice = {"source": sub_file_path, "page_number": sub_page_number}
            else:
                sub_choice = {"source": sub_file_path}

            sub_choices.append(sub_choice)

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)

            for sub_choice in sub_choices:
                icon = utils.get_source_icon(sub_choice['source'])
                if "page_number" in sub_choice:
                    st.info(f"{sub_choice['source']} (ページNo.{sub_choice['page_number']})", icon=icon)
                else:
                    st.info(f"{sub_choice['source']}", icon=icon)

        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["main_message"] = main_message
        content["main_file_path"] = main_file_path
        if "page" in llm_response["context"][0].metadata:
            content["main_page_number"] = main_page_number
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    else:
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)

        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["answer"] = ct.NO_DOC_MATCH_MESSAGE
        content["no_file_path_flg"] = True

    return content


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    st.markdown(llm_response["answer"])

    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()

        message = "情報源"
        st.markdown(f"##### {message}")

        file_path_list = []
        file_info_list = []

        for document in llm_response["context"]:
            file_path = document.metadata["source"]
            if file_path in file_path_list:
                continue

            if "page" in document.metadata:
                page_number = document.metadata["page"]
                file_info = f"{file_path} (ページNo.{page_number})"
            else:
                file_info = f"{file_path}"

            icon = utils.get_source_icon(file_path)
            st.info(file_info, icon=icon)

            file_path_list.append(file_path)
            file_info_list.append(file_info)

    content = {}
    content["mode"] = ct.ANSWER_MODE_2
    content["answer"] = llm_response["answer"]
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list

    return content