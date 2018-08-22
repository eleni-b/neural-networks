% processing input data
load dataSet
categories=sum(TrainDataTargets,2);  % sum of elements of each column
figure;
bar(categories);
title('Initial category distribution');
minimum=min(categories); % 54 samples per category
index=zeros(12,minimum);
for i=1:12;
    index(i,:)=find(TrainDataTargets(i,:),minimum);
end
  TrainSelection = TrainData(:,index);
  SelectionTargets = TrainDataTargets(:, index);
  categoriesSelection = sum(SelectionTargets,2);
  figure;
  bar(categoriesSelection);
  title('Final category distribution');
  SelectionCollumns=size(TrainSelection,2);
  permutation= randperm(SelectionCollumns);
  TrainSelection = TrainSelection(:,permutation);
  SelectionTargets = SelectionTargets(:, permutation);
  [TrainSelection,settings] = removeconstantrows(TrainSelection);
  TestData = removeconstantrows('apply',TestData,settings);
  [TrainSelection,settings] = processpca(TrainSelection,0.001);
  TestData = processpca('apply',TestData,settings);
  Result=zeros(6,3);
%DOKIMES ARXITEKTONIKIS(LAYERS/NEURONS)
  for i=1:6
  net=newff(TrainSelection,SelectionTargets,[5*i]);
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  Result(i,1)=a1;
  Result(i,2)=nanmean(a2);
  Result(i,3)=sum(a3)/12;
  h=figure;
  h=plotperform(tr);
  str=sprintf('HL%i.jpg',i);
  str1=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',Result(i,1), Result(i,2) ,Result(i,3)); 
  title(str1)
  print(h,'-djpeg',str);
  close(h);
  end
 
  Result2=zeros(6,3);
   for j=1:6
   for i=1:6
  net=newff(TrainSelection,SelectionTargets,[5*j 5*i]);
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  Result2(i,1)=a1;
  Result2(i,2)=nanmean(a2);
  Result2(i,3)=sum(a3)/12;
  h=figure;
  h=plotperform(tr);
  str=sprintf('HL%i%i.jpg',j,i);
  str1=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, Result2(i,1) ,Result2(i,3)); 
  title(str1);
  print(h,'-djpeg',str);
   close(h);
   end
   end
   
   Hidden1=25;
   Hidden2=15; %best results
   
   %change learning algorithm
   
   
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='traingdx';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str);
  print(h,'-djpeg','traingdx.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='trainlm';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str);
  print(h,'-djpeg','trainglm.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
 net.trainFcn='traingd';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str);
  print(h,'-djpeg','traingd.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='traingda';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str);
  print(h,'-djpeg','traingda.jpg');
  close(h);
  
   %change transfer functions
   
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.layers{3}.transferFcn = 'hardlim';
  net.trainFcn='trainlm';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','harldim.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
   net.trainFcn='trainlm';
  net.layers{3}.transferFcn = 'logsig';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','logsig.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='trainlm';
  net.layers{3}.transferFcn = 'tansig';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
 a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','tansig.jpg');
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='trainlm';
  net.layers{3}.transferFcn = 'purelin';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

 a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','purelin.jpeg');
   close(h);
  
  %CHANGE VALIDATION
  for i=1:10
   net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn='trainlm';
   net.divideParam.trainRatio=1;
  net.divideParam.valRatio=0.0;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=5*i;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

 a2=nanmean(a2);
 a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
   str2=sprintf('Noval%i.jpg',i);
  print(h,'-djpeg',str2);
   close(h);  
  end
  
 % Adjust learning rate
  for i=1:8
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
    net.trainParam.lr=0.05*i;
  net.trainFcn = 'traingd';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

 a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
   str1=sprintf('D%i.jpeg',i);
  print(h,'-djpeg',str1);
  close(h);
  
  net=newff(TrainSelection,SelectionTargets,[Hidden1 Hidden2]);
  net.trainFcn = 'traingdx';
  net.trainParam.lr=0.05*i;
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
   str1=sprintf('DX%i.jpeg',i);
  print(h,'-djpeg',str1);
  close(h);
  
  
  end
  
  
  %XWRIS PROEPEKSERGASIA
  clear all;
  load dataSet;
  
  
    for i=1:6
  net=newff(TrainData,TrainDataTargets,[5*i]);
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainData,TrainDataTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  Result(i,1)=a1;
  Result(i,2)=nanmean(a2);
  Result(i,3)=sum(a3)/12;
  h=figure;
  h=plotperform(tr);
  str=sprintf('NOHL%i.jpg',i);
  str1=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',Result(i,1), Result(i,2) ,Result(i,3)); 
  title(str1)
  print(h,'-djpeg',str);
  close(h);
  end
 
  Result2=zeros(6,3);
   for j=1:6
   for i=1:6
  net=newff(TrainData,TrainDataTargets,[5*j 5*i]);
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainData,TrainDataTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  Result2(i,1)=a1;
  Result2(i,2)=nanmean(a2);
  Result2(i,3)=sum(a3)/12;
  h=figure;
  h=plotperform(tr);
  str=sprintf('NOHL%i%i.jpg',j,i);
  str1=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, Result2(i,1) ,Result2(i,3)); 
  title(str1);
  print(h,'-djpeg',str);
   close(h);
   end
   end
%learngd
  net=newff(TrainSelection,SelectionTargets,[25 15],{'tansig','tansig','purelin'},'trainlm','learngd');

  net.trainFcn='trainlm';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','learngd.jpg');
  close(h);
  %learngdm
  
  net=newff(TrainSelection,SelectionTargets,[25 15],{'tansig','tansig','purelin'},'trainlm','learngdm');
  
  net.trainFcn='trainlm';
  net.divideParam.trainRatio=0.8;
  net.divideParam.valRatio=0.2;
  net.divideParam.testRatio=0;
  net.trainParam.epochs=300;
  [net,tr]=train(net,TrainSelection,SelectionTargets);
  TestDataOutput=sim(net,TestData);
  [a1,a2,a3]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
  
  a2=nanmean(a2);
  a3=sum(a3)/12;
  h=figure;
   plotperform(tr);
   str=sprintf('Accuracy = %0.4f , Mean precision = %0.4f , Mean recall = %0.4f ',a1, a2 ,a3); 
   title(str)
  print(h,'-djpeg','learngdm.jpg');
  close(h);
  
 
