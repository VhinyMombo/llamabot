import requests
import json
#import gradio as gr
import streamlit as st
from flask import Flask, request, jsonify
from flask import render_template
global url, headers

url = 'http://localhost:11434/api/generate'

headers = { 'Content-Type': 'application/json' }


# data = { 'model': 'llama2', 
#         'stream': False, 
#         'prompt': 'Who is StepHEN Hawkings ? ' }
    
# response = requests.post(url, headers=headers, data=json.dumps(data))

# if response.status_code == 200:
#     response_text = response.text
#     data = json.loads(response_text)
#     actual_response = data['response']
#     print(actual_response)
# else:
#     print(f'Error: {response.status_code, response.text}')



def generate_response(prompt):

    #data['prompt'] = prompt
    data = { 'model': 'llama2', 
    'stream': False, 
    'prompt': prompt}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data['response']
        return actual_response
    else:
        return f'Error: {response.status_code, response.text}'




# iface = gr.Interface(fn=generate_response, 
#                      in
# puts=gr.inputs.Textbox(lines=2, label='Enter your prompt here'), 
#                      outputs='text')

# iface.launch()


# app = Flask(__name__)
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     prompt = request.form.get('prompt')
#     if prompt:
#         response = generate_response(prompt)
#         return jsonify({'response': response})
#     else:
#         return jsonify({'error': 'No prompt provided'}), 400

# if __name__ == "__main__":
#     app.run()


def main():
    prompt = st.text_input('Enter your prompt here')
    if prompt:
        response = generate_response(prompt)
        st.write(response)

if __name__ == "__main__":
    main()

# app = Flask(__name__)

# @app.route('/generate', methods=['POST'])
# def generate():
#     prompt = request.json.get('prompt')
#     if prompt:
#         response = generate_response(prompt)
#         return jsonify({'response': response})
#     else:
#         return jsonify({'error': 'No prompt provided'}), 400

# if __name__ == "__main__":
#     app.run()