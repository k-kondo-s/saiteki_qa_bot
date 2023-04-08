# %%
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from saiteki_qa_agent import SaitekiQaAgent

# 環境変数から Slack の API キーを取得
try:
    SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
    SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
except:
    print('SLACK_BOT_TOKEN, SLACK_APP_TOKEN を環境変数に設定してください')
    exit()

# Slack のアプリを初期化
app = App(token=SLACK_BOT_TOKEN)
qa = SaitekiQaAgent()


def _message_builder(event, result):
    """返信するメッセージを作成する
    雰囲気以下のようなかんじ
        @k-kondo
        設備の故障を反映するには、イベントモードを使用して計画に故障を追加できます。以下の手順に従ってください。
        1. 設備軸を開き、ヘッダーのモード選択からイベントモードを選択します。
        2. ガントチャート上で故障した設備の軸上をクリックし、新規イベントを割り当てます。
        3. サイドメニューのイベント種別で非稼働を選択し、イベント名に「メンテナンス」などと入力します。期間の調整が必要な場合は開始日と終了日を設定し、最後に作成を押下します。
        4. イベントモードを終了した後、最適化を実行します。
        注意: 固定されている割り当ては、最適化の対象外となります。固定を解除してから最適化を実行することで、最適化対象になります。
        関連記事:
        - 設備が故障したので、使えない時間帯を反映したい
        - 最適化に失敗した
        - スタッフが急遽欠勤になった
        - スケジュールを手動で修正したい
        - スタッフの残業時間を設定したい
    """
    message = f'<@{event["user"]}>\n'
    message += f'{result["answer_text"]}\n\n'
    if len(result['source_documents']) > 0:
        message += '関連記事(関連度%):\n'
        for source_document in result['source_documents']:
            title, url, score = source_document['title'], source_document['url'], source_document['score']
            # score を小数点以下2桁に丸めて文字列にする
            score = str(int((round(score, 2) * 100)))
            message += f'- <{url}|{title}({score}%)>\n'
    return message


@app.event("app_mention")
def respond_to_mention(event, say):
    """chatbotにメンションが付けられたときのハンドラ
    """
    print(event)

    # 質問に対する回答を取得する
    try:
        result = qa.run(event['text'])
    except Exception as e:
        result = {'answer_text': f'エラーがおきました :しゅん: \n```{e.args}\n```',
                  'source_documents': []}

    # 返信するメッセージを作成
    message = _message_builder(event, result)
    print(message)

    # メッセージを送信。スレッドに返信する。
    thread_ts = event.get("thread_ts", None) or event["ts"]
    say(text=message, thread_ts=thread_ts, reply_broadcast=True)


@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)


# Slack のアプリを起動
SocketModeHandler(app, SLACK_APP_TOKEN).start()
