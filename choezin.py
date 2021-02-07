from flask import Flask, request, render_template,jsonify
import nltk
from autocorrect import spell
from gensim.summarization import summarize as g_sumn

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/preproc', methods=["GET"])
def pre_process():
    return render_template('preproc.html')


@app.route('/others', methods=["GET"])
def others():
    return render_template('others.html')


@app.route('/summary', methods=["GET"])
def summary():
    return render_template('text_Summarization.html')

@app.route('/lower', methods=["GET", "POST"])
def lower_case():
    text1 = request.form['text']
    word = text1.lower()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/sent_tokenize', methods=["GET", "POST"])
def sent_tokenize():
    text = request.form['text']
    sent_tokenize = nltk.sent_tokenize(text)
    result = {
        "result": str(sent_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/word_tokenize', methods=["GET", "POST"])
def word_tokenize():
    text = request.form['text']
    word_tokenize = nltk.word_tokenize(text)
    result = {
        "result": str(word_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/spell_check', methods=["GET", "POST"])
def spell_check():
    text = request.form['text']
    spells = [spell(w) for w in (nltk.word_tokenize(text))]
    result = {
        "result": " ".join(spells)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/stemming', methods=["GET", "POST"])
def stemming():
    from nltk.stem import SnowballStemmer
    snowball_stemmer = SnowballStemmer('english')

    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
    result = {
        "result": " ".join(stemmed_word)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_tags', methods=["GET", "POST"])
def remove_tags():
    import re
    text = request.form['text']
    cleaned_text = re.sub('<[^<]+?>', '', text)
    result = {
        "result": cleaned_text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/remove_numbers', methods=["GET", "POST"])
def remove_numbers():
    text = request.form['text']
    remove_num = ''.join(c for c in text if not c.isdigit())
    result = {
        "result": remove_num
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_punct', methods=["GET", "POST"])
def remove_punct():
    from string import punctuation
    def strip_punctuation(s):
        return ''.join(c for c in s if c not in punctuation)

    text = request.form['text']
    text = strip_punctuation(text)
    result = {
        "result": text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route("/summarize", methods=["GET","POST"])
def summarize():
    text = request.form['text']
    sent = nltk.sent_tokenize(text)
    if len(sent) < 4:
        summary1 =  "please pass more than 10 sentences to summarize the text"
    else:
        summary = g_sumn(text)
        summ = nltk.sent_tokenize(summary)
        summary1 = (" ".join(summ[:2]))
    result = {
        "result": summary1
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
