function [] = Jacobian_Plots(jac_mat, cols, y_var, z_var, dt, fig_dir, str_append, DT_mat)

    % colors and graphical parameters
    MyOrange = [ 231,101,45 ] ;
    MyOrange = MyOrange / 255 ;
    MyBlue   =  [ 60,106,209 ] ;
    MyBlue   = MyBlue / 255 ;
    
    DefGreen = [ 0.4660    0.6740    0.1880] ;
    OliveGreen = [53 98 26 ] / 255 ;

    colororder([MyOrange; MyBlue; DefGreen; OliveGreen])
    
    fontsize = 14 ; linewidth = 3 ; factor = 0.75 ;

    % scale of shocks
    T = size(jac_mat, 1);


    % Plot Impulse Response
    figure(1)
    plot((0:(T-1)) .* dt, jac_mat(:,cols),'LineStyle','-','LineWidth',linewidth); hold on;
    colororder([MyOrange; MyBlue; DefGreen; OliveGreen]/2)
    set(gca,'ColorOrderIndex',1)
    if (exist('DT_mat', 'var'))
        plot((0:(T-1)) .* dt, DT_mat(:,cols),'LineStyle','--','LineWidth',linewidth); hold on;
    end
    % plot((0:(T-1)), DT_mat(:,(cols-1) * dt + 1),'LineStyle','--','LineWidth',linewidth); hold on;
    title(strcat("Columns of Jacobian of ", y_var, " to ", z_var),'FontSize',fontsize)
    xlabel('Time t')
    ylabel('Jacobian Value')
    xlim([0,T*dt])
    AxisFonts(factor*fontsize,fontsize,factor*fontsize,fontsize)
    if (exist('DT_mat', 'var'))
        legend([strcat("CT s=", string((cols-1)*dt)), strcat("DT s=", string((cols-1)*dt))],'Location','southeast')
    else
        legend([strcat("CT s=", string((cols-1)*dt))],'Location','southeast')
    end
    saveas(gcf, strcat(fig_dir, 'jac_', y_var, '_', z_var, str_append, '.png'))
    hold off
