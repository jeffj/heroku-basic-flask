from flask import Flask
from datetime import datetime
import lightgbm
bst = lightgbm.Booster(model_file='saved.txt')

app = Flask(__name__)

predictVal = [[2.0000e+01, 3.0000e+00, 8.0000e+01, 1.1622e+04, 1.0000e+00,
        0.0000e+00, 3.0000e+00, 3.0000e+00, 1.0000e+00, 4.0000e+00,
        0.0000e+00, 1.2000e+01, 1.0000e+00, 2.0000e+00, 0.0000e+00,
        2.0000e+00, 5.0000e+00, 6.0000e+00, 1.9610e+03, 1.9610e+03,
        1.0000e+00, 0.0000e+00, 1.1000e+01, 1.3000e+01, 3.0000e+00,
        0.0000e+00, 3.0000e+00, 4.0000e+00, 1.0000e+00, 4.0000e+00,
        4.0000e+00, 4.0000e+00, 5.0000e+00, 4.6800e+02, 4.0000e+00,
        1.4400e+02, 2.7000e+02, 8.8200e+02, 0.0000e+00, 4.0000e+00,
        1.0000e+00, 3.0000e+00, 8.9600e+02, 0.0000e+00, 0.0000e+00,
        8.9600e+02, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
        2.0000e+00, 1.0000e+00, 4.0000e+00, 5.0000e+00, 7.0000e+00,
        0.0000e+00, 0.0000e+00, 2.0000e+00, 1.9610e+03, 3.0000e+00,
        1.0000e+00, 7.3000e+02, 4.0000e+00, 5.0000e+00, 2.0000e+00,
        1.4000e+02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.2000e+02,
        0.0000e+00, 0.0000e+00, 3.0000e+00, 0.0000e+00, 0.0000e+00,
        6.0000e+00, 2.0100e+03, 9.0000e+00, 4.0000e+00 ]]

@app.route('/')
def homepage():


    pred=bst.predict(predictVal)
    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>
    <p>I predict {prediction} till the reckoning.</p>


    <img src="http://loremflickr.com/600/400" />
    """.format(time='Whisky Time', prediction=pred[0])

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
