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
w1=normrnd(0,1,784,200)*0.01;
bias1=zeros(1,200);
hiddenoutput1=zeros(1,200);
w2=normrnd(0,1,200,50)*0.01;
bias2=zeros(1,50);
hiddenoutput2=zeros(1,50);
w3=normrnd(0,1,50,10)*0.01;
bias3=zeros(1,10);
finaloutput=zeros(1,10);
diff1=zeros(1,200);
diff2=zeros(1,50);
diff3=zeros(1,10);
learnrate3=0.01;
learnrate2=0.01;
learnrate1=0.01;
for a=1:1 %train time
   for i=1:60000 %train data
       i
       hiddenoutput1=data(i,:)*w1;
       hiddenoutput1=hiddenoutput1-bias1;
       for m=1:200
           hiddenoutput1(m)=max(0,hiddenoutput1(m));
       end
       hiddenoutput2=hiddenoutput1*w2;
       hiddenoutput2=hiddenoutput2-bias2;
       for c=1:50
           hiddenoutput2(c)=max(0,hiddenoutput2(c));
       end
       finaloutput=hiddenoutput2*w3;
       finaloutput=finaloutput-bias3;
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
           diff3(n)=diff*(result(n)-finaloutput(n));
           bias3(n)=bias3(n)-learnrate3*diff3(n);
       end
       for c=1:50
           diff=0;
           if (hiddenoutput2(c)>0)
               diff=1;
           end
           diff2(c)=w3(c,:)*diff3';
           diff2(c)=diff*diff2(c);
           bias2(c)=bias2(c)-learnrate2*diff2(c);
       end
       for m=1:200
           diff=0;
           if (hiddenoutput1(m)>0)
               diff=1;
           end
           diff1(m)=w2(m,:)*diff2';
           diff1(m)=diff*diff1(m);
           bias1(m)=bias1(m)-learnrate1*diff1(m);
           for j=1:784
               w1(j,m)=w1(j,m)+learnrate1*diff1(m)*data(i,j);
           end
       end
       for c=1:50
           for k=1:200
               w2(k,c)=w2(k,c)+learnrate2*diff2(c)*hiddenoutput1(k);
           end
       end
       for n=1:10
           for j=1:50
               w3(j,n)=w3(j,n)+learnrate3*diff3(n)*hiddenoutput2(j);
           end
       end
   end
end
%test

sum=0;
for i=1:10000
    hiddenoutput1=test(i,:)*w1;
    hiddenoutput1=hiddenoutput1-bias1;
    for m=1:200
       hiddenoutput1(m)=max(0,hiddenoutput1(m));
    end
    hiddenoutput2=hiddenoutput1*w2;
    hiddenoutput2=hiddenoutput2-bias2;
    for c=1:50
       hiddenoutput2(c)=max(0,hiddenoutput2(c));
    end
    finaloutput=hiddenoutput2*w3;
    finaloutput=finaloutput-bias3;
    for n=1:10
       finaloutput(n)=max(0,finaloutput(n));
    end
    predict=find(finaloutput==max(finaloutput))-1;
    if testlabel(i)==predict
        sum=sum+1;
    end
end
sum/10000
