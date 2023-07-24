function [fix,fixTex]=createFixDot(cfg)

fix_width=round(((tand(cfg.FixSize)*cfg.dist)*(cfg.screen.Ypix/cfg.height)));
fix_sph = createSphere(fix_width,fix_width,fix_width);
fix_fig=uint8(ones(fix_width*2-1,fix_width*2-1)*127.5);%255);

alpha_mask=uint8(zeros(fix_width*2-1,fix_width*2-1));

[columnsInImage rowsInImage] = meshgrid(1:(size(alpha_mask,1)), 1:(size(alpha_mask,2)));
centerX=fix_width;centerY=fix_width;

radius = fix_width-1;
circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radius.^2;
alpha_mask=circlePixels*255;

fix(:,:,1)=fix_fig;
fix(:,:,2)=alpha_mask;

fixTex = Screen('MakeTexture', cfg.screen.window, fix);
