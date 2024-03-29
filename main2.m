path_yolo="labels_yolo";
path_testing="labels_testing";

names_labels = dir('labels_yolo10/*.txt');
testing_labels = dir('labels_testing/*.txt');

folder = 'fp10\';
results = 'results\';


f=1;
y=960;
x=1280;
cd=0;
skips=0;
ints=0;
counter_cl=[0,0,0,0];
counter_clt=[0,0,0,0];
x=2592;
y=1944;
crops=0;
p_tot=0;
tolti=0;
imd_hd = imageDatastore("C:\Users\loand\Documents\GitHub\Datasets\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\img");
for i=1:length(testing_labels)
    fprintf('Immagine: %d\n', i);
    
    
    txtpath=fullfile(names_labels(f).folder,names_labels(f).name);
    detected_labels=readmatrix(txtpath, ...
        'Delimiter', ' ', ...
        'ConsecutiveDelimitersRule', 'join');
    
    txtpath=fullfile(testing_labels(i).folder,testing_labels(i).name);
    original_labels=readmatrix(txtpath, ...
        'Delimiter', ' ', ...
        'ConsecutiveDelimitersRule', 'join');
    n1=names_labels(f).name;
    n2=testing_labels(i).name;
    filename = strrep(n1, '.txt', '');
    detected_labels=detected_labels(:,2:5);
    detected_labels(:,1)=detected_labels(:,1)*x;
    detected_labels(:,2)=detected_labels(:,2)*y;
    detected_labels(:,3)=detected_labels(:,3)*x;
    detected_labels(:,4)=detected_labels(:,4)*y;
    detected_labels(:,1)=detected_labels(:,1)-detected_labels(:,3)/2;
    detected_labels(:,2)=detected_labels(:,2)-detected_labels(:,4)/2;
    
    
    classes=original_labels(:,1);
    original_labels=original_labels(:,2:5);
    original_labels(:,1)=original_labels(:,1)*x;
    original_labels(:,2)=original_labels(:,2)*y;
    original_labels(:,3)=original_labels(:,3)*x;
    original_labels(:,4)=original_labels(:,4)*y;
    original_labels(:,1)=original_labels(:,1)-original_labels(:,3)/2;
    original_labels(:,2)=original_labels(:,2)-original_labels(:,4)/2;
    
    
    for ccc=1:length(classes)
        counter_clt(classes(ccc)+1)=counter_clt(classes(ccc)+1)+1;
    end
    
    ths=3;
    ll=1;
    
    %disp(size(detected_labels,1));
    while ll<=size(detected_labels,1)
        
        if(((detected_labels(ll,1)-ths)<=0) || ((detected_labels(ll,2)-ths)<=0) ||...
                ((detected_labels(ll,1)+detected_labels(ll,3)+ths)>=x) ||...
                ((detected_labels(ll,2)+detected_labels(ll,4)+ths)>=y))
            %disp(detected_labels(ll,:));
            detected_labels(ll,:)=[];
            tolti=tolti+1;
            ll=ll-1;
        end
        ll=ll+1;
    end
    [img_org,info_img]=read(imd_hd);
    img=img_org;
    p_tot=p_tot+size(detected_labels,1);
    for o=1:size(original_labels,1)
        crops=crops+1;
        %img=insertShape(img,'Rectangle',original_labels(o,:),'Color','green','LineWidth',6);
        for j=1:size(detected_labels,1)
            area = rectint(original_labels(o,:),detected_labels(j,:));
            area_org=original_labels(o,3)*original_labels(o,4);
            perc=0.01;
            if area>=25 %|| area_org <=100
                
                img=insertShape(img,'Rectangle',detected_labels(j,:),'Color','blue','LineWidth',6);
                detected_labels(j,:)=[];
                ints=ints+1;
                counter_cl(classes(o)+1)=counter_cl(classes(o)+1)+1;
                break;
            end
        end
    end
    imwrite(img,fullfile(results,strcat(filename,".jpg")));
    
    f=f+1;
    need=0;
    img=img_org;
    if need==1
        
        for o=1:size(detected_labels,1)
            crops=crops+1;
            area=0;
            for j=1:size(original_labels,1)
                area =area+ rectint(original_labels(j,:),detected_labels(o,:));
            end
            if area<=10
                fprintf('Immagine fp: %d\n',f);
                img=insertShape(img,'Rectangle',detected_labels(o,:),'Color','blue','LineWidth',6);
                imwrite(img,fullfile(folder,strcat(filename,".jpg")));
                %fermati/0;
            end
        end
        
    end
end