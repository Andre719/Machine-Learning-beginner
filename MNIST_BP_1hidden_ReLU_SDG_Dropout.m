clear all;
close all;
clc;

fp =fopen('train-images-idx3-ubyte','r');
f=fread(fp, 4,'int32', 0, 'ieee-be');
data=fread(fp,[784,60000]);
data=(data')&1;
fclose(fp);
fp =fopen('train-labels-idx1-ubyte','r');
f=fread(fp, 2,'int32', 0, 'ieee-be');
label=fread(fp,60000);
fclose(fp);
fp =fopen('t10k-images-idx3-ubyte','r');
f=fread(fp, 4,'int32', 0, 'ieee-be');
test=fread(fp,[784,10000]);
test=(test')&1;
fclose(fp);
fp =fopen('t10k-labels-idx1-ubyte','r');
f=fread(fp, 2,'int32', 0, 'ieee-be');
testlabel=fread(fp,10000);
fclose(fp);

%BP (one hidden 100 points)
w1=normrnd(0,1,784,100)*0.01;
bias1=zeros(1,100);
hiddenoutput=zeros(1,100);
w2=normrnd(0,1,100,10)*0.01;
bias2=zeros(1,10);
finaloutput=zeros(1,10);
diff1=zeros(1,100);
diff2=zeros(1,10);
learnrate2=0.01;
learnrate1=0.01;
for a=1:1 %train time
   for i=1:10000 %train data
       i
       hiddenoutput=data(i,:)*w1;
       hiddenoutput=hiddenoutput-bias1;
       for m=1:100
           if (rand<0.5)
               hiddenoutput(m)=max(0,hiddenoutput(m))/0.5;
           else
               hiddenoutput(m)=0;
           end
       end
       finaloutput=hiddenoutput*w2;
       finaloutput=finaloutput-bias2;
       for n=1:10
           finaloutput(n)=max(0,finaloutput(n));
       end
       result=zeros(1,10);
       result(label(i)+1)=1;
       for n=1:10
           diff=0;
           if (finaloutput(n)>0)
               diff=1;
           end
           diff2(n)=diff*(result(n)-finaloutput(n));
           bias2(n)=bias2(n)-learnrate2*diff2(n);
       end
       for m=1:100
           diff=0;
           if (hiddenoutput(m)>0)
               diff=1;
           end
           diff1(m)=w2(m,:)*diff2';
           diff1(m)=diff*diff1(m);
           bias1(m)=bias1(m)-learnrate1*diff1(m);
           for j=1:784
               w1(j,m)=w1(j,m)+learnrate1*diff1(m)*data(i,j);
           end
       end
       for n=1:10
           for k=1:100
               w2(k,n)=w2(k,n)+learnrate2*diff2(n)*hiddenoutput(k);
           end
       end
   end
end
%test

sum=0;
for i=1:10000
    hiddenoutput=test(i,:)*w1;
    hiddenoutput=hiddenoutput-bias1;
    for m=1:100
        hiddenoutput(m)=max(0,hiddenoutput(m));
    end
    finaloutput=hiddenoutput*w2;
    finaloutput=finaloutput-bias2;
    for n=1:10
        finaloutput(n)=max(0,finaloutput(n));
    end
    predict=find(finaloutput==max(finaloutput))-1;
    if testlabel(i)==predict
        sum=sum+1;
    end
end
sum/10000
