function [  ] = plot_iteration( i, x ,y )
    figure(i+1)
    plot(x,y,'-o');
    axis equal
    title(['iteration ' num2str(i)])
end

