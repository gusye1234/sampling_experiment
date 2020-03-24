## Playground

**LOG**

```shell
code
├── TrainProcedure.py # the training and testing function 
├── advsampler.py # not used for now
├── dataloader.py # explained by its name
├── main.py
├── model.py # store models
├── utils.py # loss function, sampling methods, metrics
├── register.py # register models and train procedure here
└── world.py # some global vars
```

```shell
optional arguments:
  -h, --help            show this help message and exit
  --recdim RECDIM       the embedding size of recmodel
  --vardim VARDIM       the embedding size of varmodel
  --reclr RECLR         learning rate for rec model
  --varlr VARLR         learning rate for var model
  --xdecay XDECAY       weight decay for var model
  --vardecay VARDECAY   weight decay for var model
  --recdecay RECDECAY   weight decay for rec model
  --layer LAYER         the layer num of lightGCN
  --dropout DROPOUT     using the dropout or not
  --keepprob KEEPPROB   the batch size for bpr loss training procedure
  --testbatch TESTBATCH
                        the batch size of users for testing
  --ontest ONTEST       set 1 to run test on test1.txt, set 0 to run test on
                        validation.txt
  --path PATH           path to save weights
  --tensorboard TENSORBOARD
                        enable tensorboard
  --comment COMMENT
  --load LOAD
  --epochs EPOCHS
  --vartype VARTYPE     var model types
  --sampletype SAMPLETYPE
                        sampling methods types
```



