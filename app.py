from flask import Flask, render_template, request, send_file
import tensorflow as tf
import os

app = Flask(__name__)

model = tf.keras.models.load_model('models/CNN-256-128-128-0')

app.config["UPLOAD_DIR"] = "uploads"
@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        for file in request.files.getlist('file'):
             path = os.path.join(app.config['UPLOAD_DIR'], file.filename)
             file.save(path)
             send_file(path, as_attachment=True)
        return render_template("upload.html", msg = "Files uplaoded successfully.", link = f'{request.base_url}/download/{file.filename}')

    return render_template("upload.html", msg = "")

@app.route('/download/<path:filename>')
def downloadFile (filename):
    path = os.path.join(app.config['UPLOAD_DIR'], filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run()