from flask import Flask, render_template, request
import tensorflow as tf
import os

app = Flask(__name__)

model = tf.keras.models.load_model('models/CNN-256-128-128-0')

app.config["UPLOAD_DIR"] = "uploads"
@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        for file in request.files.getlist('file'):
             file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
        return render_template("upload.html", msg = "Files uplaoded successfully.")

    return render_template("upload.html", msg = "")

if __name__ == "__main__":
    app.run()