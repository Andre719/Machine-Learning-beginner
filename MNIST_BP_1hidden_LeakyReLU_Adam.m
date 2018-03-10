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
learnrate2=0.001;
learnrate1=0.001;
muw1=zeros(784,100);
mub1=zeros(1,100);
muw2=zeros(100,10);
mub2=zeros(1,10);
gradw1=zeros(784,100);
gradb1=zeros(1,100);
gradw2=zeros(100,10);
gradb2=zeros(1,10);
friction=0.9;
decayrate=0.995;
for a=1:1 %train time
   for i=1:60000 %train data
       i
       hiddenoutput=data(i,:)*w1;
       hiddenoutput=hiddenoutput-bias1;
       for m=1:100
           hiddenoutput(m)=max(0.01*hiddenoutput(m),hiddenoutput(m));
       end
       finaloutput=hiddenoutput*w2;
       finaloutput=finaloutput-bias2;
       for n=1:10
           finaloutput(n)=max(0.01*finaloutput(n),finaloutput(n));
       end
       result=zeros(1,10);
       result(label(i)+1)=1;
       for n=1:10
           diff=0.01;
           if (finaloutput(n)>0)
               diff=1;
           end
           diff2(n)=diff*(result(n)-finaloutput(n));
           mub2(n)=friction*mub2(n)+(1-friction)*diff2(n);
           gradb2(n)=decayrate*gradb2(n)+(1-decayrate)*(diff2(n))*(diff2(n));
           first_unbias=mub2(n)/(1-friction^i);
           second_unbias=gradb2(n)/(1-decayrate^i);
           bias2(n)=bias2(n)-learnrate2*first_unbias/(sqrt(second_unbias)+1e-7);
       end
       for m=1:100
           diff=0.01;
           if (hiddenoutput(m)>0)
               diff=1;
           end
           diff1(m)=w2(m,:)*diff2';
           diff1(m)=diff*diff1(m);
           mub1(m)=friction*mub1(m)+(1-friction)*diff1(m);
           gradb1(m)=decayrate*gradb1(m)+(1-decayrate)*(diff1(m))*(diff1(m));
           first_unbias=mub1(m)/(1-friction^i);
           second_unbias=gradb1(m)/(1-decayrate^i);
           bias1(m)=bias1(m)-learnrate1*first_unbias/(sqrt(second_unbias)+1e-7);
           for j=1:784
               muw1(j,m)=friction*muw1(j,m)+(1-friction)*diff1(m)*data(i,j);
               gradw1(j,m)=decayrate*gradw1(j,m)+(1-decayrate)*(diff1(m)*data(i,j))*(diff1(m)*data(i,j));
               first_unbias=muw1(j,m)/(1-friction^i);
               second_unbias=gradw1(j,m)/(1-decayrate^i);
               w1(j,m)=w1(j,m)+learnrate1*first_unbias/(sqrt(second_unbias)+1e-7);
           end
       end
       for n=1:10
           for k=1:100
               muw2(k,n)=friction*muw2(k,n)+(1-friction)*diff2(n)*hiddenoutput(k);
               gradw2(k,n)=decayrate*gradw2(k,n)+(1-decayrate)*(diff2(n)*hiddenoutput(k))*(diff2(n)*hiddenoutput(k));
               first_unbias=muw2(k,n)/(1-friction^i);
               second_unbias=gradw2(k,n)/(1-decayrate^i);
               w2(k,n)=w2(k,n)+learnrate2*first_unbias/(sqrt(second_unbias)+1e-7);
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
        hiddenoutput(m)=max(0.01*hiddenoutput(m),hiddenoutput(m));
    end
    finaloutput=hiddenoutput*w2;
    finaloutput=finaloutput-bias2;
    for n=1:10
        finaloutput(n)=max(0.01*finaloutput(n),finaloutput(n));
    end
    predict=find(finaloutput==max(finaloutput))-1;
    if testlabel(i)==predict
        sum=sum+1;
    end
end
sum/10000
