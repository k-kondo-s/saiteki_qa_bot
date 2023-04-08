# %%
import os
import time
import urllib.parse

import pinecone
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone


class VectorStore():
    """
    VectorStore クラスは、Pinecone を使用してドキュメントのベクトル化とベクトルストアへの保存を処理します。

    主なメソッド:
    - delete_all_indexes: すべてのインデックスを削除します。
    - store_to_vectoredb: ドキュメントをベクトル化し、ベクトルストアに保存します。
    """

    def __init__(self):
        """
        VectorStore インスタンスを初期化します。環境変数から Pinecone の API キーと環境を取得し、Pinecone を初期化します。
        """
        # 環境変数から Pinecone の API キーと環境を取得
        try:
            self.pinecone_api_key = os.environ['PINECONE_API_KEY']
            self.pinecone_env = os.environ['PINECONE_ENV']
            self.pinecone_index_name = os.environ['PINECONE_INDEX_NAME']
        except:
            raise Exception(
                'PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME を環境変数に設定してください')

        # pinecone を初期化
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_env,
        )

    def delete_all_indexes(self):
        """
        すべてのインデックスを削除します。
        """
        # すべてのインデックスを削除
        index = pinecone.Index(self.pinecone_index_name)
        index.delete(delete_all=True)

    def store_to_vectoredb(self, documents):
        """
        documents をベクトル化し、ベクトルストアに保存します。

        Args:
            documents (list): Document オブジェクトのリスト
        """
        # documents をベクトル化して vectore store に保存する
        embeddings = OpenAIEmbeddings()
        Pinecone.from_documents(documents, embeddings,
                                index_name=self.pinecone_index_name)
        print('Done.')


class SaitekiManualHandler():
    """
    SaitekiManualHandler クラスは、Saiteki サポートサイトから記事を取得し、ドキュメントを生成、分割する機能を提供します。

    主なメソッド:
    - generate_document: 与えられた URL からドキュメントを生成します。
    - split_documents: ドキュメントを分割します。
    - get_documents_from_urls: 与えられた URL からドキュメントを取得します。
    """

    def __init__(self) -> None:
        """
        SaitekiManualHandler インスタンスを初期化します。
        """
        self.saiteki_url_parts = urllib.parse.urlparse(
            'https://support.saiteki.works')

    def _get_url_by_path(self, path):
        """
        path から URL を取得します。

        Args:
            path (str): URL のパス部分

        Returns:
            str: 完全な URL
        """
        # path から url を取得する
        parts = self.saiteki_url_parts._replace(path=path)
        return parts.geturl()

    def _request(self, url):
        """
        url にアクセスし、取得した HTML を解析して BeautifulSoup オブジェクトを返します。

        Args:
            url (str): アクセスする URL

        Returns:
            BeautifulSoup: 解析された HTML の BeautifulSoup オブジェクト
        """
        # url にアクセスする
        # headers は zendesk が定めるものを使う
        print(f'Accessing {url} ...')
        res = requests.get(
            url=url,
            headers={'user-agent': 'Zendesk/External-Content'}
        )
        soup = BeautifulSoup(res.text, "html.parser")
        # DDos対策
        time.sleep(1)
        return soup

    def _get_page_urls(self, url):
        """
        与えられた URL から、関連するページの URL リストを取得します。

        Args:
            url (str): 基本となる URL

        Returns:
            list: 関連するページの URL リスト
        """

        # 与えられた url にアクセス
        soup = self._request(url)
        # その中で、食わせたいページの path を取得する
        paths = []
        for h2 in soup.find_all('h2'):
            paths.append(h2.find('a').get('href'))
        # path から url を取得する
        url_list = [self._get_url_by_path(path) for path in paths]
        return url_list

    def __get_article_urls(self, url):
        """
        与えられた URL から、記事の URL リストを取得します。

        Args:
            url (str): 基本となる URL

        Returns:
            list: 記事の URL リスト
        """
        # 与えられた url にアクセス
        soup = self._request(url)
        # class 名が article-list-item の a タグの href を取得する
        paths = []
        for a in soup.find_all('a', class_='article-list-link'):
            paths.append(a.get('href'))
        # path から url を取得する
        url_list = [self._get_url_by_path(path) for path in paths]
        return url_list

    def generate_document(self, url):
        """
        与えられた URL にアクセスし、ページタイトルと本文を取得して Document オブジェクトを生成します。

        Args:
            url (str): 記事の URL

        Returns:
            Document: 生成された Document オブジェクト
        """
        # 与えられた url にアクセス
        soup = self._request(url)
        # ページタイトルを取得。'工程デザイナーご利用の流れ – saiteki.works サポートサイト'
        # という形式なので、' – ' で分割して、先頭の要素を取得する
        title = soup.find('title').text.split(' – ')[0]
        # ページ本文から document を作る
        metadata = {"source": url, 'title': title}
        document = Document(page_content=soup.text, metadata=metadata)
        return document

    def _generate_documents(self, page_urls):
        """
        与えられたページの URL リストから Document オブジェクトのリストを生成します。

        Args:
            page_urls (list): ページの URL リスト

        Returns:
            list: Document オブジェクトのリスト
        """
        print(f'Generating documents ...')
        documents = []
        for url in page_urls:
            documents.append(self.generate_document(url))
        return documents

    def split_documents(self, documents):
        """
        文書を分割します。

        Args:
            documents (list): Document オブジェクトのリスト

        Returns:
            list: 分割された Document オブジェクトのリスト
        """
        print(f'Splitting documents ...')
        # documents を分割する
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20,
            length_function=len,
        )
        splitted_documents = splitter.split_documents(documents)
        return splitted_documents

    def get_documents_from_urls(self, urls):
        """
        与えられた URL リストから、関連するページと記事の URL を取得し、それらのドキュメントを生成、分割して返します。

        Args:
            urls (list): URL のリスト

        Returns:
            list: Document オブジェクトのリスト
        """
        # urls それぞれから、食わせたいページの url を取得する
        content_urls = []
        for url in urls:
            part_page_urls = self._get_page_urls(url)
            # この下のページに興味があるので、それを取得する
            for part_url in part_page_urls:
                content_urls += self.__get_article_urls(part_url)

        # documents を作る
        documents = self._generate_documents(content_urls)

        # documents を分割する
        # splitted_documents = self.split_documents(documents)

        return documents


if __name__ == '__main__':

    # 与えた url から順に辿って、食わせたいページの document を取得する
    # これらのページはサポートページトップの「最適ワークス」「サービスマネージャー」「よくある質問(FAQ)」のページ
    root_urls = [
        'https://support.saiteki.works/hc/ja/categories/8436793810329-%E6%9C%80%E9%81%A9%E3%83%AF%E3%83%BC%E3%82%AF%E3%82%B9',
        'https://support.saiteki.works/hc/ja/categories/8436803690009-%E3%82%B5%E3%83%BC%E3%83%93%E3%82%B9%E3%83%9E%E3%83%8D%E3%83%BC%E3%82%B8%E3%83%A3%E3%83%BC',
        'https://support.saiteki.works/hc/ja/categories/900000309486-%E3%82%88%E3%81%8F%E3%81%82%E3%82%8B%E3%81%94%E8%B3%AA%E5%95%8F-FAQ-'
    ]
    handler = SaitekiManualHandler()
    documents = handler.get_documents_from_urls(root_urls)

    # vector store の中を全部削除したあと、
    # document をベクトル化して vectore store に保存する
    store = VectorStore()
    store.delete_all_indexes()
    store.store_to_vectoredb(documents)
