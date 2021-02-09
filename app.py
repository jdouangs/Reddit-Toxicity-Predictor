# import necessary libraries
from flask_pymongo import PyMongo
import reddit_scrape
from flask import Flask, render_template, jsonify, request, redirect

app = Flask(__name__)
app.config["MONGO_URI"] = 'mongodb://@ds125453.mlab.com:25453/redditanalysis'
mongo = PyMongo(app)

@app.route("/home", methods = ['GET', 'POST'])
def home():
    reddit_data = mongo.db.redditdata.find()
    parameter = 'Toxic Prediction'
    filter_text = 'All'
    comment_length = reddit_data.count()
    if request.method == 'POST':
        if request.form['toxic-filter'] == '':
            reddit_data = mongo.db.redditdata.find()
            parameter = 'Toxic Prediction'
            comment_length = reddit_data.count()
        else:
            filter_text = request.form['toxic-filter']
            parameter = str(filter_text) + " Prediction"
            reddit_data = mongo.db.redditdata.find({parameter: 1})
            comment_length = reddit_data.count()
    return render_template("index.html", reddit_data = reddit_data, parameter = parameter, comment_length = comment_length, filter_text = filter_text)

@app.route("/scrape", methods = ['POST'])
def scrape():
    if request.method == 'POST':
        choice = request.form['subreddit-scrape']
    mongo.db.redditdata.drop()
    scraped_data = reddit_scrape.scrape(choice)
    master = scraped_data[0]
    mongo.db.redditdata.insert_many(master)
    return redirect("/home", code=302)

if __name__ == '__main__':
    app.run(debug=True)
