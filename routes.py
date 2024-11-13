import json

from flask import Blueprint, jsonify, request, send_file, make_response
import os
import csv
import requests
from openai import OpenAI

UPLOAD_DIR = "/home/mt798jx/uploads"
RESULT_DIR = "/home/mt798jx/results"

bp = Blueprint('main', __name__)

headers1 = None
student_answers = {}

@bp.route('/upload', methods=['POST'])
def handle_file_upload():
    if 'file' not in request.files:
        return jsonify({"message": "No files part in the request"}), 400

    files = request.files.getlist('file')
    uploaded_files = []
    errors = []

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    for file in files:
        if file.content_type != 'text/csv':
            errors.append(f"{file.filename} is not a CSV file")
            continue

        try:
            upload_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(upload_path)
            uploaded_files.append(file.filename)
        except Exception as e:
            print(e)
            errors.append(f"Failed to upload {file.filename}")

    if uploaded_files:
        message = {"message": f"Files uploaded successfully: {uploaded_files}"}
        if errors:
            message["errors"] = errors
        return jsonify(message), 200
    else:
        return jsonify({"message": "No files were uploaded", "errors": errors}), 400

@bp.route('/list', methods=['GET'])
def list_uploaded_files():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_names = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.csv')]

    if not file_names:
        return jsonify({"message": "No CSV files found in the uploads directory."}), 200

    return jsonify(file_names), 200

@bp.route('/generatedlist', methods=['GET'])
def list_generated_csv_files():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    file_names = [f for f in os.listdir(RESULT_DIR) if f.lower().endswith('.csv')]

    if not file_names:
        return jsonify({"message": "No CSV files found in the results directory."}), 200

    return jsonify(file_names), 200

@bp.route('/text', methods=['GET'])
def list_generated_files():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    file_names = [f for f in os.listdir(RESULT_DIR) if f.lower().endswith('.txt')]

    if not file_names:
        return jsonify({"message": "No TXT files found in the results directory."}), 200

    return jsonify(file_names), 200

@bp.route('/delete', methods=['DELETE'])
def delete_file():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify({"error": "No fileName provided"}), 400

    file_path = os.path.join(UPLOAD_DIR, file_name)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"message": f"File deleted successfully: {file_name}"}), 200
        else:
            return jsonify({"error": "File does not exist"}), 404
    except Exception as e:
        return jsonify({"error": "An error occurred during file deletion"}), 500

@bp.route('/download', methods=['GET'])
def download_file():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify({"error": "No fileName provided"}), 400

    file_path = os.path.join(RESULT_DIR, file_name)

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 404

    try:
        return send_file(
            file_path,
            as_attachment=True,
            download_name=file_name,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        print(e)
        return jsonify({"error": "An error occurred during file download"}), 500

@bp.route('/previewtext', methods=['GET'])
def preview_txt_file():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify({"error": "No fileName provided"}), 400

    file_path = os.path.join(RESULT_DIR, file_name)

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 404

    try:
        with open(file_path, "r") as file:
            content = file.read()
        return make_response(content, 200)
    except Exception as e:
        print(e)
        return jsonify({"error": "An error reading file"}), 500


@bp.route('/create', methods=['GET'])
def create_file():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify(["No fileName provided"]), 400

    input_file_path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(input_file_path):
        return jsonify(["Súbor nenájdený"]), 400

    result_chunks = []
    base_name = os.path.splitext(file_name)[0]
    new_file_name = f"{base_name}.csv"
    result_file_path = os.path.join(RESULT_DIR, new_file_name)

    try:
        with open(input_file_path, "r") as input_file, open(result_file_path, "w", newline='') as result_file:
            csv_writer = csv.writer(result_file)
            csv_writer.writerow(["Id", "Skóre", "Odpoveď", "Stručný komentár"])

            current_answer = []
            for line in input_file:
                if line.startswith("---"):
                    if current_answer:
                        create_answer("".join(current_answer), csv_writer, result_chunks)
                        current_answer = []
                    line = line[3:]
                current_answer.append(line)

            if current_answer:
                create_answer("".join(current_answer), csv_writer, result_chunks)

    except Exception as e:
        print(e)
        return jsonify(["Failed to process file"]), 500

    return jsonify(result_chunks), 200


def create_answer(answer_block, csv_writer, result_chunks):
    arr = answer_block.split("//", 3)
    csv_row = [arr[i].strip() if i < len(arr) else "" for i in range(4)]
    csv_writer.writerow(csv_row)
    result_chunks.append(",".join(csv_row))


@bp.route('/preview', methods=['GET'])
def preview_file():
    global headers1, student_answers
    file_name = request.args.get("fileName")
    if not file_name:
        return make_response("No fileName provided", 400)

    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        return make_response("File not found", 400)

    student_answers = {}

    try:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file)
            headers1 = next(csv_reader, None)
            headers2 = next(csv_reader, None)

            for row in csv_reader:
                try:
                    user_id = int(row[0].strip())
                    answer = row[1].strip().replace("\"", "")
                    rating = float(row[2].strip().replace(",", "."))

                    student_answers[user_id] = {
                        "answer": answer,
                        "rating": rating
                    }

                except (IndexError, ValueError) as e:
                    print(f"Error processing row: {row}, Error: {e}")

        preview_content = []
        preview_content.append("Headers: " + ", ".join(headers1) if headers1 else "No headers found")
        preview_content.append("Headers: " + ", ".join(headers2) if headers2 else "No headers found")

        for i, (user_id, data) in enumerate(student_answers.items()):
            preview_content.append(f"ID: {user_id}, Answer: {data['answer']}, Score: {data['rating']}")

        preview_text = "\n".join(preview_content)
        return make_response(preview_text, 200)

    except Exception as e:
        print(f"Error reading file: {e}")
        return make_response("Error reading file", 500)

MODEL = "gemini-1.5-flash"
API_KEY = "AIzaSyBLMXPfd4I2cly2GhJAVrYPG-3a1EKYVL8"

SYSTEM_MESSAGE = (
    "[system] „Ako učiteľ hodnotiaci odpovede študentov z predmetu Operačné systémy, prosím, hodnotťte každú odpoveď systematicky na základe nasledujúcich kritérií. Zadajte skóre od 0 do 100, kde:\n" +
            "\n" +
            "0 bodov znamená, že odpoveď je úplne nesprávna a neobsahuje žiadne relevantné informácie alebo kľúčové slová.\n" +
            "50 bodov naznačuje, že odpoveď obsahuje čiastočne správne informácie, ale chýbajú kľúčové prvky alebo je významne nepresná.\n" +
            "100 bodov znamená, že odpoveď je úplne správna, zahŕňa všetky kľúčové slová a pojmy a plne zodpovedá zadaniu otázky.\n" +
            "Pre každé hodnotenie dodržte presnú štruktúru a poskytnite tieto údaje:\n" +
            "\n" +
            "Skóre – uvedené ako Integer číslo (môže byť napríklad 56, 77, 84).\n" +
            "ID študenta – unikátny identifikátor, aby bolo hodnotenie priradené správnej osobe.\n" +
            "Odpoveď študenta – presný, nezmenený text odpovede, aby mohol byť odoslaný na platformu Moodle na ďalšie spracovanie.\n" +
            "Krátky komentár v slovenčine – stručné zhrnutie hodnotenia, ktoré študentovi poskytne konštruktívnu spätnú väzbu.\n" +
            "Použite túto štruktúru bez prázdnych riadkov medzi hodnoteniami:\n" +
            "\n" +
            "---12345//83.5//[študentova celá odpoveď]//[krátka poznámka]\n" +
            "Ako hodnotiť odpovede:\n" +
            "\n" +
            "Identifikácia kľúčových slov a konceptov: Pri hodnotení dbajte na dôležité pojmy a kľúčové slová uvedené v odpovedi. Skontrolujte, či študent správne vysvetlil dôležité aspekty, aj keď niektoré body mohli byť vyjadrené implicitne alebo inými slovami.\n" +
            "Úprava skóre na základe kontextu: Ak niektoré časti odpovede naznačujú správnu odpoveď, no nie sú plne rozvinuté alebo obsahujú menšie nepresnosti, upravte skóre primerane. Vždy zohľadnite kontext otázky a odpovede.\n" +
            "Normalizácia hodnotenia: V rámci hodnotenia ignorujte diakritiku, veľkosť písmen a zbytočné medzery, aby bolo zabezpečené spravodlivé a konzistentné hodnotenie medzi všetkými študentmi.\n" +
            "Komentáre k hodnoteniu: Pri každej odpovedi poskytnite krátky komentár v slovenčine, ktorý objasní dôvod prideleného skóre, napríklad: „Správna odpoveď, obsahuje všetky dôležité pojmy.“ alebo „Čiastočná odpoveď, chýba vysvetlenie hlavného konceptu.“\n" +
            "Konzistencia a férovosť hodnotenia: Pri hodnotení podobných odpovedí dbajte na konzistenciu, aby sa hodnotenie nemenilo pre podobné alebo identické odpovede. Poskytnite jasnú spätnú väzbu v zmysle rovnakých pravidiel pre všetkých študentov.\n" +
            "Ukážka odpovede:\n" +
            "\n" +
            "---12345//83.5//Operačné systémy sú základom softvéru, ktorý riadi hardvérové zdroje.//Správne, avšak stručné, chýba vysvetlenie procesného manažmentu.\n" +
            "---67890//100//Operačný systém riadi hardvér počítača a poskytuje platformu pre aplikácie.//Kompletná odpoveď s presnými detailmi, výborné.\n" +
            "Ak študent neodpovedal - ukážka odpovede:\n" +
            "\n " +
            "---10678//-1//null//chýba odpoveď"
)

def create_prompt(batch):
    prompt = f"{headers1}\n"
    for user_id, answer in batch.items():
        prompt += f"User ID {user_id}: Answer: {answer}\n"
    return prompt

def evaluate_with_gemini(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_MESSAGE + "\n\n" + prompt}]
            }
        ],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code: {response.status_code}")


@bp.route('/process-gemini', methods=['GET'])
def process_file_gemini():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify({"error": "File name not provided"}), 400

    if headers1 is None or not student_answers:
        return jsonify({"error": "Headers or student answers not loaded"}), 400

    question = headers1
    batch_size = 25
    results = []
    max_result_size = 300000
    result_chunks = []

    student_entries = list(student_answers.items())
    for start_index in range(0, len(student_entries), batch_size):
        batch = dict(student_entries[start_index:start_index + batch_size])

        prompt = create_prompt(batch)
        try:
            evaluation = evaluate_with_gemini(prompt)
            results.append(evaluation["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if sum(len(r) for r in results) > max_result_size:
            result_chunks.append("\n".join(results))
            results = []

    if results:
        result_chunks.append("\n".join(results))

    output_file_path = os.path.join(RESULT_DIR, f"{os.path.splitext(file_name)[0]}-results.txt")
    with open(output_file_path, "w") as file:
        for chunk in result_chunks:
            file.write(chunk)

    return jsonify(result_chunks), 200

client = OpenAI(api_key="sk-proj-FrY06JjHaOzQZjTejQUeT3BlbkFJ6RG42qREbT5suj8K6bvw")

def process_chatgpt_response(response):
    results = []

    for tool_call in response.choices[0].message.tool_calls:

        arguments = json.loads(tool_call.function.arguments)

        student_id = arguments["student_id"]
        score = arguments["score"]
        student_answer = arguments["student_answer"]
        comment = arguments["comment"]

        result_string = f"---{student_id}//{score}//{student_answer}//{comment}"
        results.append(result_string)
    final_output = "\n".join(results)
    return final_output

def evaluate_with_chatgpt(prompt):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.6,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "evaluate_student_answers",
                        "description": "Hodnotenie odpovedí študentov z predmetu Operačné systémy na základe stanovených kritérií.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "student_id": {"type": "string"},
                                "score": {"type": "integer"},
                                "student_answer": {"type": "string"},
                                "comment": {"type": "string"}
                            },
                            "required": ["student_id", "score", "student_answer", "comment"]
                        }
                    }
                }
            ],
            parallel_tool_calls=True,
            response_format={"type": "text"}
        )


        content = response
        response_text = process_chatgpt_response(content)
        return response_text
    except Exception as e:
        raise Exception(f"API request failed: {e}")

@bp.route('/process-chatgpt', methods=['GET'])
def process_file_chatgpt():
    file_name = request.args.get("fileName")
    if not file_name:
        return jsonify({"error": "File name not provided"}), 400

    if headers1 is None or not student_answers:
        return jsonify({"error": "Headers or student answers not loaded"}), 400

    batch_size = 5
    results = []
    max_result_size = 300000
    result_chunks = []

    student_entries = list(student_answers.items())
    for start_index in range(0, len(student_entries), batch_size):
        batch = dict(student_entries[start_index:start_index + batch_size])

        prompt = create_prompt(batch)
        print(prompt)
        try:
            evaluation = evaluate_with_chatgpt(prompt)
            results.append(evaluation)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if sum(len(r) for r in results) > max_result_size:
            result_chunks.append("\n".join(results))
            results = []

    if results:
        result_chunks.append("\n".join(results))

    output_file_path = os.path.join(RESULT_DIR, f"{os.path.splitext(file_name)[0]}-results.txt")
    with open(output_file_path, "w") as file:
        for chunk in result_chunks:
            file.write(chunk)

    return jsonify(result_chunks), 200


@bp.route("/compare-scores", methods=["GET"])
def compare_scores():
    file_name = request.args.get("fileName")

    response_map = {}
    uploads_score_distribution = create_empty_score_distribution()
    results_score_distribution = create_empty_score_distribution()

    uploads_file_exists = False
    results_file_exists = False

    uploads_file_name = file_name.replace("-results", "")
    uploads_file_path = os.path.join(UPLOAD_DIR, uploads_file_name)
    results_file_path = os.path.join(RESULT_DIR, file_name)

    if os.path.exists(uploads_file_path):
        calculate_uploads_score_distribution(uploads_file_path, uploads_score_distribution)
        uploads_file_exists = True

    if os.path.exists(results_file_path):
        calculate_results_score_distribution(results_file_path, results_score_distribution)
        results_file_exists = True

    if not uploads_file_exists and not results_file_exists:
        return jsonify({"error": "Both upload and result files not found"}), 400

    if uploads_file_exists:
        response_map["uploadsScoreDistribution"] = uploads_score_distribution
    if results_file_exists:
        response_map["resultsScoreDistribution"] = results_score_distribution

    return jsonify(response_map)


def create_empty_score_distribution():
    return {
        "-1-49": 0,
        "50-59": 0,
        "60-69": 0,
        "70-79": 0,
        "80-89": 0,
        "90-100": 0,
    }


def calculate_uploads_score_distribution(file_path, score_distribution):
    processed_student_ids = set()

    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                student_id = int(row[0].strip())
                score = float(row[2].strip()) * 100
                if student_id not in processed_student_ids:
                    processed_student_ids.add(student_id)
                    calculate_score_range(score, score_distribution)
            except (ValueError, IndexError) as e:
                print(f"Error processing row {row}: {e}")


def calculate_results_score_distribution(file_path, score_distribution):
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            if len(row) >= 2:
                try:
                    score = float(row[1].strip())
                    calculate_score_range(score, score_distribution)
                except ValueError as e:
                    print(f"Error processing row {row}: {e}")


def calculate_score_range(score, score_distribution):
    if -1000 <= score <= 50:
        score_distribution["-1-49"] += 1
    elif 51 <= score <= 60:
        score_distribution["50-59"] += 1
    elif 61 <= score <= 70:
        score_distribution["60-69"] += 1
    elif 71 <= score <= 80:
        score_distribution["70-79"] += 1
    elif 81 <= score <= 90:
        score_distribution["80-89"] += 1
    elif 91 <= score <= 100:
        score_distribution["90-100"] += 1

####