function [] = IRFs_fn(irf, y_var, z_var, dt, fig_dir, str_append, DT_irf)

    % colors and graphical parameters
    MyOrange = [ 231,101,45 ] ;
    MyOrange = MyOrange / 255 ;
    MyBlue   =  [ 60,106,209 ] ;
    MyBlue   = MyBlue / 255 ;
    
    DefGreen = [ 0.4660    0.6740    0.1880] ;
    OliveGreen = [53 98 26 ] / 255 ;
    
    fontsize = 14 ; linewidth = 3 ; factor = 0.75 ;

    % scale of shocks
    T = length(irf);

    % Plot Impulse Response
    figure(1)
    plot((0:(T-1)) .* dt, irf,'Color',MyOrange,'LineStyle','-','LineWidth',linewidth); hold on;
    if (exist('DT_irf', 'var'))
        plot((0:(T-1)), DT_irf(1:T),'Color',MyOrange/2,'LineStyle',':','LineWidth',linewidth); hold on;
    end
    % plot((0:(T-1)), DT_irf(1:T),'Color',MyOrange/2,'LineStyle',':','LineWidth',linewidth); hold on;
    % plot(1:n.Ntot, ss.g .* n.vdaa(:) ./ sum(ss.g .* n.vdaa(:)),'Color',MyOrange,'LineStyle','-','LineWidth',linewidth); hold on;
    title(strcat("Impulse Response of ", y_var, " to ", z_var),'FontSize',fontsize)
    xlabel('Time')
    ylabel('% Deviation from SS')
    xlim([0,T*dt])
    AxisFonts(factor*fontsize,fontsize,factor*fontsize,fontsize)
    if (exist('DT_irf', 'var'))
        legend('Cont Time', 'Discrete Time')
    else
        legend('Cont Time')
    end
    saveas(gcf, strcat(fig_dir, 'irf_', y_var, '_', z_var, str_append, '.png'))
    hold off


