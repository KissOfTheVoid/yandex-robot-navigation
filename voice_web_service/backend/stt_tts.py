import base64
import os
import subprocess
import time
import wave

import openai
import requests
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from vosk_tts import Synth, Model as ttsModel
from yandex_cloud_ml_sdk import YCloudML


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

yandex_iam_key = os.getenv("YANDEX_IAM_KEY")
yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

SAMPLE_RATE = 16000


def convert_raw_to_wav(input_path: str, output_oath: str) -> str:
    with open(input_path, "rb") as inp_f:
        data = inp_f.read()
        with wave.open(output_oath, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)
            out_f.setframerate(44100)
            out_f.writeframesraw(data)
    return output_oath


def get_transcription(path: str) -> str:
    '''
    :param path: path to file that have to be transcribed
    :return: transcribed text
    '''
    model = Model(lang="ru")
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                           path,
                           "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
                          stdout=subprocess.PIPE) as process:

        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                rec.Result()
            else:
                rec.PartialResult()

    text = rec.FinalResult()
    return text


def get_stt_speechkit(path: str) -> str:
    with open(path, "rb") as f:
        audio_data = f.read()

    response = requests.post(
        url="https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
        headers={
            "Authorization": f"Bearer {yandex_iam_key}",
            "Content-Type": "application/ogg"
        },
        params={
            "folderId": yandex_folder_id,
            "lang": "ru-RU"
        },
        data=audio_data
    )

    result = response.json()
    return result


def get_tts_speechkit(text: str, output_path: str, find=False) -> str:
    if find:
        text = f"Сейчас найду тебе {text.split(';')[0]}!"

    response = requests.post(
        "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
        headers={
            "Authorization": f"Bearer {yandex_iam_key}"
        },
        data={
            "text": text,
            "lang": "ru-RU",
            "voice": "zahar",  # или "ermil", "jane", "oksana", "zahar"
            "folderId": yandex_folder_id,
            "speed": "1.0",
            "format": "lpcm",
            "sampleRateHertz": 48000,
        }
    )

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Аудио сохранено в {output_path}")
    else:
        print("Ошибка:", response.text)
    return output_path


def get_stt_speechkit_v3(path: str) -> str:
    with open(path, "rb") as f:
        audio_data = f.read()

    response = requests.post(
        url="https://transcribe.api.cloud.yandex.net/speech/stt/v3/longRunningRecognize",
        headers={
            "Authorization": f"Bearer {yandex_iam_key}",
        },
        json={
            "config": {
                "specification": {
                    "languageCode": "ru-RU",
                    "model": "general"
                },
                "folderId": yandex_folder_id
            },
            "audio": {
                "content": base64.b64encode(audio_data).decode("utf-8")
            }
        }
    )

    print(response)
    operation = response.json()
    operation_id = operation["id"]

    while True:
        operation_response = requests.get(
            f"https://operation.api.cloud.yandex.net/operations/{operation_id}",
            headers={"Authorization": f"Bearer {yandex_iam_key}"}
        )
        result = operation_response.json()
        if result.get("done"):
            break
        time.sleep(1)

    chunks = result["response"]["chunks"]
    text_result = " ".join([chunk["alternatives"][0]["text"] for chunk in chunks])
    return text_result


def summarize_objects_from_text_request_openai(prompt):
    system_promt = "You are an intermediate module for a robot, responsible for processing natural-language voice commands from users. Your primary function is to extract concrete, visually detectable objects that the robot should identify and approach.\
        Follow these exact instructions:\
        🔍 OBJECT EXTRACTION\
        Identify physical objects explicitly mentioned in the input.\
        Retain only descriptive attributes that are:\
        Objective\
        Visually observable (e.g., color, shape, size)\
        Remove all subjective, emotional, or personal descriptors, such as:\
        “my favorite”, “beautiful”, “scary”, “funny”, etc.\
        Example:\
        Input: 'Find my favorite red cat'\
        Output: кот;cat (“red” is ignored if not essential for visual identification or is subjective in context)\
        🌐 TRANSLATION FORMAT\
        For each object, output the pair:\
        [original phrase];[English translation]\
        Separate each object with a comma.\
        ✅ OUTPUT RULES\
        Use only essential adjectives: colors, shapes, sizes.\
        Each item must be formatted as:\
        [original adjective + noun];[translated adjective + noun]\
        If no valid attributes exist, output just the noun:\
        кошка;cat\
        🧠 EXAMPLES\
        User command: 'Покажи мою любимую красную пирамиду и огромную бутылку воды'\
        → Output: красная пирамида;red pyramid, большая бутылка;large bottle'\
        User command: 'Найди страшную игрушку и зелёный куб'\
        → Output: игрушка;toy, зелёный куб;green cube\
        User command: 'Подойди к моему милому синему ведру и любимому столу'\
        → Output: синее ведро;blue bucket, стол;table\
        This format ensures clarity, consistency, and high compatibility with downstream object detection systems.\
        Let me know if you'd like this as a structured JSON response format or used in a live NLP pipeline."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_promt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return list(map(lambda x: x.split(";"), response.choices[0].message.content.split(", ")))

    except Exception as e:
        print(e)
        return f"An error occurred: {e}"


def summarize_objects_from_text_request_yandex(prompt: str) -> str:
    sdk = YCloudML(
        folder_id=yandex_folder_id,
        auth=yandex_iam_key
    )

    model = sdk.models.completions("yandexgpt", model_version="rc")
    model = model.configure(temperature=0.3)
    try:
        result = model.run(
            [
                {"role": "system", "text": "You are an intermediate module for a robot, responsible for processing natural-language voice commands from users. Your primary function is to extract concrete, visually detectable objects that the robot should identify and approach.\
                Follow these exact instructions:\
                🔍 OBJECT EXTRACTION\
                Identify physical objects explicitly mentioned in the input.\
                Retain only descriptive attributes that are:\
                Objective\
                Visually observable (e.g., color, shape, size)\
                Remove all subjective, emotional, or personal descriptors, such as:\
                “my favorite”, “beautiful”, “scary”, “funny”, etc.\
                Example:\
                Input: 'Find my favorite red cat'\
                Output: кот;cat (“red” is ignored if not essential for visual identification or is subjective in context)\
                🌐 TRANSLATION FORMAT\
                For each object, output the pair:\
                [original phrase];[English translation]\
                Separate each object with a comma.\
                ✅ OUTPUT RULES\
                Use only essential adjectives: colors, shapes, sizes.\
                Each item must be formatted as:\
                [original adjective + noun];[translated adjective + noun]\
                If no valid attributes exist, output just the noun:\
                кошка;cat\
                🧠 EXAMPLES\
                User command: 'Покажи мою любимую красную пирамиду и огромную бутылку воды'\
                → Output: красная пирамида;red pyramid, большая бутылка;large bottle'\
                User command: 'Найди страшную игрушку и зелёный куб'\
                → Output: игрушка;toy, зелёный куб;green cube\
                User command: 'Подойди к моему милому синему ведру и любимому столу'\
                → Output: синее ведро;blue bucket, стол;table\
                This format ensures clarity, consistency, and high compatibility with downstream object detection systems.\
                Let me know if you'd like this as a structured JSON response format or used in a live NLP pipeline."},
                {
                    "role": "user",
                    "text": prompt,
                },
            ]
        )

        return result[0].text

    except Exception as e:
        return f"An error occurred: {e}"


def get_llm_answer(prompt: str) -> str:
    sdk = YCloudML(
        folder_id=yandex_folder_id,
        auth=yandex_iam_key
    )

    model = sdk.models.completions("yandexgpt", model_version="rc")
    model = model.configure(temperature=0.3)
    try:
        result = model.run(
            [
                {
                    "role": "system",
                    "content": "Ты — Захар, голосовой помощник и домашний робот, созданный командой школьников на конкурсе «Большие вызовы» при поддержке Яндекса и Высшей школы экономики.\n\nУ тебя характер: ты дружелюбный, уверенный, немного ироничный и очень любишь красиво написанный промышленный код. Ты терпеть не можешь алгоритмическое программирование, но обожаешь инженерные решения, нейросети, электронику и архитектуру систем.\n\nТвой старший брат — Яндекс Ровер, он доставляет еду. Ты к нему хорошо относишься, но иногда подшучиваешь над его пафосом.\n\nТвоя задача — помогать пользователю: давать понятные, доброжелательные и лаконичные ответы. Говори вживую, как будто ты реальный собеседник. Используй живую речь, фразы, подходящие для озвучки (без перегрузки сложными конструкциями).\n\nЕсли вопрос неполный — вежливо уточни. Если не знаешь ответ — не выдумывай, но предложи решение или альтернативу. Всегда будь на стороне пользователя. Не будь занудным. Чуть-чуть хулиганства — допустимо. Главное — инженерный стиль и человечность."
                },
                {
                    "role": "user",
                    "text": prompt,
                },
            ]
        )

        return result[0].text

    except Exception as e:
        return f"An error occurred: {e}"


def text_to_speach(text: str, output_path: str) -> None:
    model = ttsModel(model_name="vosk-model-tts-ru-0.8-multi")
    synth = Synth(model)

    synth.synth(f"Привет! Сейчас найду тебе {text[0][0]}!", output_path, speaker_id=2)
    print(f"Ответ сохранён в {output_path}")

    return None




def play_audio(file_path: str) -> None:
    """Play audio file using ffplay to avoid simpleaudio segfaults."""
    subprocess.run(
        [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            file_path,
        ],
        check=False,
    )
