##importing library
library(h2o)
#initialize h2o
h2o.init()
##set working directory
setwd("D:\\rushikesh\\H2O_Coursera")
data=read.csv("cacao.882.csv")
dim(data)
## check any NA
sum(is.na(data))
##converting data into h2o frame
data.h2o=as.h2o(data)
##splitind data into train and test
split_sets<-h2o.splitFrame(data.h2o,.75,seed = 12345)
print(dim(split_sets[[1]]))
print(dim(split_sets[[2]]))
##creating train and test dataset
train.h2o<-split_sets[[1]]
test.h2o<-split_sets[[2]]
colnames(train.h2o)
# dependent variable
# independent variable list
y.dep<-6
x.ind<-c(1:5,7:ncol(train.h2o))

system.time(base_model<-h2o.deeplearning(x=x.ind,
                             y=y.dep,
                             training_frame = train.h2o,
                             nfolds = 5,
                             seed = 12345))
base_model
h2o.performance(base_model)
## performance on test dataset 

h2o.performance(base_model,test.h2o)
## plot
plot(base_model)


####tunning Apporoach
m_100_epoch<-h2o.deeplearning(x=x.ind,
                              y=y.dep,
                              training_frame =train.h2o,
                              nfolds = 5,
                              epochs = 150,
                              stopping_metric = "logloss",#default
                              stopping_rounds = 5, #dfault
                              seed = 12345
                                                          )

h2o.performance(m_100_epoch)

h2o.performance(m_100_epoch,test.h2o)
plot(m_100_epoch)

#### increasing layer and epoch 

m_200<-h2o.deeplearning(x=x.ind,
                        y=y.dep,
                        training_frame = train.h2o,
                        nfold=5,
                        epochs = 200,
                        hidden = c(200,200,200),
                        seed=12345)

h2o.performance(m_200)

h2o.performance(m_200,test.h2o)

plot(m_200)


#######  Reducing the epoch and changing layer
m_100_400<-h2o.deeplearning(x=x.ind,
                        y=y.dep,
                        training_frame = train.h2o,
                        nfolds = 5,
                        epochs = 100,
                        hidden = c(400,400),
                        seed=12345
)

m_100_400

h2o.performance(m_100_400)
h2o.performance(m_100_400,test.h2o)
plot(m_100_400)

#### best model is m_100_400 comparing the logloss
## Model saved m_100_400
h2o.saveModel(m_100_400,path=getwd(),force = T)
##load the saved model
h2o.loadModel("D:\\rushikesh\\H2O_Coursera\\DeepLearning_model_R_1567493887969_4")
### save the base model
h2o.saveModel(base_model,path=getwd(),force = T)
###load the base Model
h2o.loadModel( "D:\\rushikesh\\H2O_Coursera\\DeepLearning_model_R_1567496654280_1")
h2o.shutdown()

