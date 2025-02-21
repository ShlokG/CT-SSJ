function [  ] = AxisFonts( xTickSize , xLabelSize , yTickSize , yLabelSize )

% x axis
xl = get(gca,'XLabel');
xAX = get(gca,'XAxis');
set(xAX,'FontSize', xTickSize )
set(xl, 'FontSize', xLabelSize);

% y axis
yl = get(gca,'YLabel');
yAY = get(gca,'YAxis');
set(yAY,'FontSize', yTickSize )
set(yl, 'FontSize', yLabelSize);

end

