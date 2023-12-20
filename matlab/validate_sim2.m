

load fwe_simulation2.out
load bwe_simulation2.out
load heun_simulation2.out

t = fwe_simulation2(:,1);
fwe = fwe_simulation2(:,2);
bwe = bwe_simulation2(:,26);
heun = heun_simulation2(:,51);

f1 = zeros(50001,1);
f2 = zeros(50001,1);
f3 = zeros(50001,1);
for i=1:50001
    f1(i) = sqrt(1/(10000 + 20*t(i))) ;
    f2(i) = 2.4 - sqrt(1/(1/(2.15^2)+20*t(i)));
    f3(i) = 4.9 - sqrt(1/(1/(4.4^2)+20*t(i)));
end



figure(1)
semilogx(t,fwe,'*')
hold on 
semilogx(t,f1)

figure(2)
semilogx(t,bwe,'*')
hold on 
semilogx(t,f2)

figure(3)
semilogx(t,heun,'*')
hold on 
semilogx(t,f3)

