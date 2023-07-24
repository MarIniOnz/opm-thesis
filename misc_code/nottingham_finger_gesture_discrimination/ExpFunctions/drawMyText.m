function drawMyText (cfg, window, Fix_pos, fix_tex, msg1, msg2, h_shift, v_shift, colour)

    Screen('TextSize',window, round(cfg.screen.Xpix/60));
    %Screen('DrawTexture', window, fix_tex, [], Fix_pos);
    DrawFormattedText(window,msg1,'center',Fix_pos(2)-v_shift,colour);
    DrawFormattedText(window, msg2 ,'center',Fix_pos(2)+v_shift, colour);

end


