# %%
import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone


class SaitekiQaAgent():
    """
    SaitekiQaAgent は、質問応答 (QA) システムを実装したクラスです。これは、質問を受け取り、関連する情報を検索し、最適な回答を生成します。
    Pinecone をベクトルストアとして使用し、環境変数から Pinecone の API キーと環境を取得します。

    主なメソッド:
    - initialize_chain: Pinecone を初期化し、QA チェーンを作成します。
    - run: 質問に対する回答を取得します。必要に応じて QA チェーンを初期化します。

    使用例:
    qa_agent = SaitekiQaAgent()
    result = qa_agent.run("質問のテキスト")
    """

    def __init__(self) -> None:
        """
        SaitekiQaAgent インスタンスを初期化します。qa_chain は None に設定されます。
        """
        self.qa_chain = None

    def initialize_chain(self):
        """
        Pinecone を初期化し、QA チェーンを作成します。環境変数から Pinecone の API キーと環境を取得し、
        それらを使用して Pinecone を初期化します。その後、ベクトルストアを用いた検索機能を持つ QA チェーンを作成します。
        """
        # vectore store として pinecone を使用
        embeddings = OpenAIEmbeddings()

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
        print(self.pinecone_index_name)
        vectorstore = Pinecone.from_existing_index(
            index_name=self.pinecone_index_name, embedding=embeddings)

        # qa chain を作成
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-4', temperature=0),
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
        )
        self.qa_chain = qa_chain

    def run(self, prompt):
        """
        質問に対する回答を取得します。QA チェーンが初期化されていない場合、initialize_chain メソッドを使用して初期化します。
        その後、質問を QA チェーンに渡し、回答と関連する情報を含む結果を返します。

        Args:
            prompt (str): 質問のテキスト

        Returns:
            dict: 回答テキストと関連情報を含む辞書
        """
        # qa chain がなければ作成
        if self.qa_chain is None:
            self.initialize_chain()

        # 質問に対する回答を取得する
        result = {}
        answer = self.qa_chain(prompt)
        result['answer_text'] = answer['result']
        result['source_documents'] = [
            {'title': source_doc.metadata['title'],
                'url': source_doc.metadata['source']}
            for source_doc in answer['source_documents']
        ]
        return result


if __name__ == '__main__':
    saiteki_qa_agent = SaitekiQaAgent()
    answer = saiteki_qa_agent.run('特急オーダーを入れたい')
    print(answer)

# %%
