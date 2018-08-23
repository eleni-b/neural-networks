%LOAD DATA TO Datae

Dmin=min(Datae.'',2);
Dmax=max(Datae,'',2);
minMax=[Dmin,Dmax]
gridsize=[10 10] 
setOrderLR = 0.9;
setOrderSteps = 250;
setTuneLR = 0.1;
somCreate(minMax,gridsize);
somTrainParameters(setOrderLR,setOrderSteps,setTuneLR);

somTrain(Datae);
figure; plot2DSomData(Patterns); %2d visualisation

