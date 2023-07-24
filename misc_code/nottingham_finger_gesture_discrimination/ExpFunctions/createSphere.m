%function to calculate a circle
function [sph] = createSphere(x,y,radius)
sph=[];
for a=(x-radius):(x+radius)
  for b=(y-radius):(y+radius)
    if pdist([x,y;a,b],'euclidean')<radius
      sph = [sph ; a,b];
    end
  end
end
sph=unique(sph,'rows');
end
