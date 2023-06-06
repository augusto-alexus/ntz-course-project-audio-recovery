from flask import Flask, render_template, request, send_file, jsonify
import os
from model import run

app = Flask(__name__)

app.config["UPLOAD_DIR"] = "uploads"
@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_DIR'], file.filename)
        print(path, file)
        file.save(path)
        filename = run(path)
        return jsonify({"link": f'{request.base_url}/download/{filename}'})

    return render_template("upload.html", msg = "")

@app.route('/download/<path:filename>')
def downloadFile (filename):
    path = os.path.join('tmp', filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run()