load heun_para_est_h.out
load observations1.txt

x = observations1(:,1);

for i=1:5
    figure(i)
    plot(x,heun_para_est_h(:,i),'b-');
    hold on
    plot(x,observations1(:,i),'k*');
end
