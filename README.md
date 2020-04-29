# tensorflow-in-practice
Coursera Specialization TensorFlow in Practice

# [TensorFlow in practice specialization](https://www.coursera.org/specializations/tensorflow-in-practice)

## What I learned

- Best practices for TensorFlow, a popular open-source machine learning framework to train a neural network for a computer vision application

- Handle real-world image data and explore strategies to prevent overfitting, including augmentation and dropout.

- Build natural language processing systems using TensorFlow.

- Apply RNNs, GRUs, and LSTMs as you train them using text repositories.

About the specialization

In this four-course Specialization, I explored exciting opportunities for AI applications. Begin by developing an understanding of how to build and train neural networks. Improve a network’s performance using convolutions as you train it to identify real-world images. I taught machines to understand, analyze, and respond to human speech with natural language processing systems. Learn to process text, represent sentences as vectors, and input data to a neural network. I even trained an AI to create original poetry!

## The specialization consists of 4 courses

### *Course 1*: **[Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://github.com/Philhoels/tensorflow-in-practice/tree/master/TensorFlow%20in%20Practice%20Specialization/Course%201%20-%20Introduction%20to%20TensorFlow%20for%20Artificial%20Intelligence%2C%20Machine%20Learning%2C%20and%20Deep%20Learning)**

#### What I learned

- Learn best practices for using TensorFlow, a popular open-source machine learning framework

- Build a basic neural network in TensorFlow

- Train a neural network for a computer vision application

- Understand how to use convolutions to improve your neural network


**Week 1**:

A New Programming Paradigm

Welcome to this course on going from Basics to Mastery of TensorFlow. We're excited you're here! In week 1 you'll get a soft introduction to what Machine Learning and Deep Learning are, and how they offer you a new programming paradigm, giving you a new set of tools to open previously unexplored scenarios. All you need to know is some very basic programming skills, and you'll pick the rest up as you go along.

You'll be working with code that works well across both TensorFlow 1.x and the TensorFlow 2.0 alpha. To get started, check out the first video, a conversation between Andrew and Laurence that sets the theme for what you'll study...

**Week 2**:

Introduction to Computer Vision

Welcome to week 2 of the course! In week 1 you learned all about how Machine Learning and Deep Learning is a new programming paradigm. This week you’re going to take that to the next level by beginning to solve problems of computer vision with just a few lines of code!

Check out this conversation between Laurence and Andrew where they discuss it and introduce you to Computer Vision!

**Week 3**:

Enhancing Vision with Convolutional Neural Networks

Welcome to week 3! In week 2 you saw a basic Neural Network for Computer Vision. It did the job nicely, but it was a little naive in its approach. This week we’ll see how to make it better, as discussed by Laurence and Andrew here.

**Week 4**:

Using Real-world Images

Last week you saw how to improve the results from your deep neural network using convolutions. It was a good start, but the data you used was very basic. What happens when your images are larger, or if the features aren’t always in the same place? Andrew and Laurence discuss this to prepare you for what you’ll learn this week: handling complex images!

### *Course 2*: **[Convolutional Neural Networks in TensorFlow](https://github.com/Philhoels/tensorflow-in-practice/tree/master/TensorFlow%20in%20Practice%20Specialization/Course%202%20-%20Convolutional%20Neural%20Networks%20in%20TensorFlow)**

#### What I learned

- Handle real-world image data

- Plot loss and accuracy

- Explore strategies to prevent overfitting, including augmentation and dropout

- Learn transfer learning and how learned features can be extracted from models

**Week 1**: 

Exploring a Larger Dataset

In the first course in this specialization, you had an introduction to TensorFlow, and how, with its high level APIs you could do basic image classification, an you learned a little bit about Convolutional Neural Networks (ConvNets). In this course you'll go deeper into using ConvNets will real-world data, and learn about techniques that you can use to improve your ConvNet performance, particularly when doing image classification!

In Week 1, this week, you'll get started by looking at a much larger dataset than you've been using thus far: The Cats and Dogs dataset which had been a Kaggle Challenge in image classification!

**Week 2**: 

Augmentation: A technique to avoid overfitting

You've heard the term overfitting a number of times to this point. Overfitting is simply the concept of being over specialized in training -- namely that your model is very good at classifying what it is trained for, but not so good at classifying things that it hasn't seen. In order to generalize your model more effectively, you will of course need a greater breadth of samples to train it on. That's not always possible, but a nice potential shortcut to this is Image Augmentation, where you tweak the training set to potentially increase the diversity of subjects it covers. You'll learn all about that this week!

**Week 3**: 

Transfer Learning

Building models for yourself is great, and can be very powerful. But, as you've seen, you can be limited by the data you have on hand. Not everybody has access to massive datasets or the compute power that's needed to train them effectively. Transfer learning can help solve this -- where people with models trained on large datasets train them, so that you can either use them directly, or, you can use the features that they have learned and apply them to your scenario. This is Transfer learning, and you'll look into that this week!

**Week 4**: 

Multiclass Classifications

You've come a long way, Congratulations! One more thing to do before we move off of ConvNets to the next module, and that's to go beyond binary classification. Each of the examples you've done so far involved classifying one thing or another -- horse or human, cat or dog. When moving beyond binary into Categorical classification there are some coding considerations you need to take into account. You'll look at them this week!

### *Course 3*: **[Natural Language Processing in TensorFlow](https://github.com/Philhoels/tensorflow-in-practice/tree/master/TensorFlow%20in%20Practice%20Specialization/Course%203%20-%20Natural%20Language%20Processing%20in%20TensorFlow)**

#### What I learned

- Build natural language processing systems using TensorFlow

- Process text, including tokenization and representing sentences as vectors

- Apply RNNs, GRUs, and LSTMs in TensorFlow

- Train LSTMs on existing text to create original poetry and more

**Week 1**: 

Sentiment in text

The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. This is the process of converting the text into numeric values, with a number representing a word or a character. This week you'll learn about the Tokenizer and pad_sequences APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!

**Week 2**: 

Word Embeddings

Last week you saw how to use the Tokenizer to prepare your text to be used by a neural network by converting words into numeric tokens, and sequencing sentences from these tokens. This week you'll learn about Embeddings, where these tokens are mapped as vectors in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space. This will begin the process of training a neural network to udnerstand sentiment in text -- and you'll begin by looking at movie reviews, training a neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those meanings.

**Week 3**: 

Sequence models

In the last couple of weeks you looked first at Tokenizing words to get numeric values from them, and then using Embeddings to group words of similar meaning depending on how they were labelled. This gave you a good, but rough, sentiment analysis -- words such as 'fun' and 'entertaining' might show up in a positive movie review, and 'boring' and 'dull' might show up in a negative one. But sentiment can also be determined by the sequence in which words appear. For example, you could have 'not fun', which of course is the opposite of 'fun'. This week you'll start digging into a variety of model formats that are used in training models to understand context in sequence!

**Week 4**: 

Sequence models and literature

Taking everything that you've learned in training a neural network based on NLP, we thought it might be a bit of fun to turn the tables away from classification and use your knowledge for prediction. Given a body of words, you could conceivably predict the word most likely to follow a given word or phrase, and once you've done that, to do it again, and again. With that in mind, this week you'll build a poetry generator. It's trained with the lyrics from traditional Irish songs, and can be used to produce beautiful-sounding verse of it's own!


### *Course 4*: **[Sequences, Time Series and Prediction](https://github.com/Philhoels/tensorflow-in-practice/tree/master/TensorFlow%20in%20Practice%20Specialization/Course%204%20-%20Sequences%2C%20Time%20Series%2C%20Prediction)**

#### What I learned

- Solve time series and forecasting problems in TensorFlow

- Prepare data for time series learning using best practices

- Explore how RNNs and ConvNets can be used for predictions

- Build a sunspot prediction model using real-world data


**Week 1**: 

Sequences and Prediction

Hi Learners and welcome to this course on sequences and prediction! In this course we'll take a look at some of the unique considerations involved when handling sequential time series data -- where values change over time, like the temperature on a particular day, or the number of visitors to your web site. We'll discuss various methodologies for predicting future values in these time series, building on what you've learned in previous courses!

**Week 2**: 

Deep Neural Networks for Time Series

Having explored time series and some of the common attributes of time series such as trend and seasonality, and then having used statistical methods for projection, let's now begin to teach neural networks to recognize and predict on time series!

**Week 3**: 

Recurrent Neural Networks for Time Series

Recurrent Neural networks and Long Short Term Memory networks are really useful to classify and predict on sequential data. This week we'll explore using them with time series...

**Week 4**: 

Real-world time series data

On top of DNNs and RNNs, let's also add convolutions, and then put it all together using a real-world data series -- one which measures sunspot activity over hundreds of years, and see if we can predict using it.
