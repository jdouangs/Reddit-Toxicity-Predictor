var x = [5,7,10];
console.log(x)

// let reddit_sentiment = Object.values(reddit_data[Sentiment])
for (var i = 0; i < sentiment.length; i ++) {
	x.push(sentiment[i])
}

var trace = {
    x: x,
    type: 'histogram',
  };
var data = [trace];
Plotly.newPlot('myDiv', data);