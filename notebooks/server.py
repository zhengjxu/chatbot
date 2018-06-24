from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from pretrained_model_predict import *
import tensorflow as tf


print('load model...')
g = tf.Graph()
# need to specify graph explicitly, otherwise there will be error when predicting
with g.as_default():
    gunthercox_word_glove_chat_bot = enc_dec_lstm()
gunthercox_word_glove_chat_bot_conversations = []


app = Flask(__name__)
# app.config.from_object(__name__)  # load config from this file , flaskr.py

# # Load default config and override config from an environment variable
# app.config.from_envvar('FLASKR_SETTINGS', silent=True)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return 'Zhengjie Xu'

@app.route('/test')
def test():
    print('test2...')
    gunthercox_word_glove_chat_bot.test_run()
    
    print('input: ', 'how are you?')
    output_sent = gunthercox_word_glove_chat_bot.guess('how are you?')
    print('reply: ', output_sent)
    print('-------')

@app.route('/gunthercox_word_glove_reply', methods=['POST', 'GET'])
def gunthercox_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            # need to specify graph explicitly, otherwise there will be error when predicting
            with g.as_default():
                reply = gunthercox_word_glove_chat_bot.guess(sent)
            gunthercox_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_glove_reply.html', conversations=gunthercox_word_glove_chat_bot_conversations)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    # need to specify graph explicitly, otherwise there will be error when predicting
    with g.as_default():
        print('test...')
        gunthercox_word_glove_chat_bot.test_run()

        print('input: ', 'how are you?')
        output_sent = gunthercox_word_glove_chat_bot.guess('how are you?')
        print('reply: ', output_sent)
        print('-------')

    print('start server...')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
