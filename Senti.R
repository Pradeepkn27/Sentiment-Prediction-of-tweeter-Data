install.packages("text2vec")
install.packages("tm")
install.packages("RWeka")
install.packages("slam")
install.packages("wordcloud")
install.packages("tokenizers")
install.packages("ggplot2")
install.packages("e1071")
install.packages("RTextTools")
install.packages("class")

#--------------------------------------------------------#
# Step 0 - Assign Library & define functions             #
#--------------------------------------------------------#

library(text2vec)
library(data.table)
library(stringr)
library(tm)
library(RWeka)
library(tokenizers)
library(slam)
library(wordcloud)
library(ggplot2)
library(graphics)
library(twitteR)
library(RTextTools)
library(e1071)
library(class)


#--------------------------------------------------------#
# Step 1 - Reading tweets data                             #
#--------------------------------------------------------#

tweets = read.csv('E:\\Pradeep_pc_backup\\Pradeep_pc_backup\\Data_Science_Material\\Specialization_project\\Tweets.csv')
tweets = twListToDF(tweets)    # Convert from list to dataframe
tweets.df_all = tweets[,c(2,6,11)]
tweets.df = tweets.df_all[,c(3)]
head(tweets.df)
tweets.df = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweets.df);head(tweets.df) 

tweets.df = gsub("@\\w+", "", tweets.df);head(tweets.df) # regex for removing @user
tweets.df = gsub("[[:punct:]]", "", tweets.df);head(tweets.df) # regex for removing punctuation mark
tweets.df = gsub("[[:digit:]]", "", tweets.df);head(tweets.df) # regex for removing numbers
tweets.df = gsub("http\\w+", "", tweets.df);head(tweets.df) # regex for removing links
tweets.df = gsub("\n", " ", tweets.df);head(tweets.df)  ## regex for removing new line (\n)
tweets.df = gsub("[ \t]{2,}", " ", tweets.df);head(tweets.df) ## regex for removing two blank space
tweets.df =  gsub("[^[:alnum:]///' ]", " ", tweets.df)     # keep only alpha numeric 
tweets.df =  iconv(tweets.df, "latin1", "ASCII", sub="")   # Keep only ASCII characters
tweets.df = gsub("^\\s+|\\s+$", "", tweets.df);head(tweets.df)  # Remove leading and trailing white space
tweets.df_all[,3]    = tweets.df # save in Data frame




head(tweets.df_all)

tweets_sentiment_virg  = tweets.df_all[tweets.df_all$airline == "Virgin America",]
tweets_sentiment_virg
tweets_sentiment_amer  = tweets[tweets$airline == "American",]
tweets_sentiment_delt  = tweets[tweets$airline == "Delta",]
tweets_sentiment_sthwe = tweets[tweets$airline == "Southwest",]
tweets_sentiment_unit  = tweets[tweets$airline == "United",]
tweets_sentiment_USAw  = tweets[tweets$airline == "US Airways",]


data = data.frame(id = 1:length(tweets_sentiment_virg$text),airline_sentiment = tweets_sentiment_virg$airline_sentiment,text = tweets_sentiment_virg$text, stringsAsFactors = F)


text.clean = function(x)                    # text data
{ require("tm")
  x  =  gsub("<.*?>", " ", x)               # regex for removing HTML tags
  x  =  iconv(x, "latin1", "ASCII", sub="") # Keep only ASCII characters
  x  =  gsub("[^[:alnum:]]", " ", x)        # keep only alpha numeric 
  x  =  tolower(x)                          # convert to lower case characters
  x  =  removeNumbers(x)                    # removing numbers
  x  =  stripWhitespace(x)                  # removing white space
  x  =  gsub("^\\s+|\\s+$", "", x)          # remove leading and trailing white space
  return(x)
}

stpw1 = readLines('E:\\Pradeep_pc_backup\\Pradeep_pc_backup\\Data_Science_Material\\Week_8\\attachment_stopwords.txt')
stpw2 = tm::stopwords('english')                   # tm package stop word list; tokenizer package has the same name function
comn  = unique(c(stpw1, stpw2))                 # Union of two list
stopwords = unique(gsub("'"," ",comn))  # final stop word lsit after removing punctuation

x  = text.clean(data$text)             # pre-process text corpus
x  =  removeWords(x,stopwords)            # removing stopwords created above
x  =  stripWhitespace(x)                  # removing white space
x  =  stemDocument(x)


require(tokenizers)
tok_fun = word_tokenizer

it_m = itoken(x,
              # preprocessor = text.clean,
              tokenizer = tok_fun,
              ids = data$id,
              progressbar = T)

vocab = create_vocabulary(it_m
                          # ngram = c(2L, 2L),
                          #stopwords = stopwords
)

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 1)
# doc_proportion_max = 0.5,
# doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_m  = create_dtm(it_m, vectorizer)
dim(dtm_m)

dtm = as.DocumentTermMatrix(dtm_m, weighting = weightTf)
#a0 = (apply(dtm, 1, sum) > 0)   # build vector to identify non-empty docs
#dtm = dtm[a0,]                  # drop empty docs

dtm_mat = as.matrix(dtm)

dtm_df = data.frame(dtm_mat,airline_sentiment = tweets_sentiment_virg$airline_sentiment,stringASFactors = FALSE)
dtm_df
# Sample the data into training and test data
sample.ind <- sample(2, 
                     nrow(dtm_df),
                     replace = T,
                     prob = c(0.6,0.4))

dtm_df.data.dev <- dtm_df[sample.ind==1,]
dtm_df.data.val <- dtm_df[sample.ind==2,]
nrow(dtm_df.data.dev)
classifier = naiveBayes(airline_sentiment ~ .,data = dtm_df.data.dev)
summary(classifier)
print(classifier)


dtm_df.data.val_without_airsent = subset(dtm_df.data.val,select = -c(dtm_df.data.val$airline_sentiment))
# test the validity
predicted = predict(classifier, dtm_df.data.val_without_airsent )
predicted
table(predicted,dtm_df.data.val$airline_sentiment)

