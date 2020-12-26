from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint
import os
import connexion
PORT = int(os.environ.get('PORT', 8080))
app = connexion.App(__name__, port = PORT, specification_dir='./')

app.add_api('models.yaml')

if __name__ == "__main__":
    app.run(debug = True)
