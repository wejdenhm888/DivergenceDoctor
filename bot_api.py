from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


from flask import Flask, request, jsonify
from flask_cors import CORS

from classify_single_change import check_divergence

app = Flask(__name__)
CORS(app)  # allow the popup/extension (different origin) to call this

REQUEST_TIMEOUT_SECONDS = 600 # overall budget for fetch + classify + recommend


@app.route("/check_divergence", methods=["GET"])
def check_divergence_endpoint():
    change_number = request.args.get("change_number", "").strip()
    if not change_number:
        return jsonify({"error": "missing change_number parameter"}), 400

    import concurrent.futures

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(check_divergence, change_number)
            result = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        return jsonify({
            "error": f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s "
                     f"(Gerrit fetch or LLM call took too long)"
        }), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
