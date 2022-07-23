addpath('C:\Users\loand\Google Drive\Ricerca\Codes\MATLAB\Utilities');

imgsDir = 'C:\Users\loand\Documents\GitHub\Blood\malaria_detection_yolo\detection10\crops';

modelsDir = 'models';
models = dir(fullfile( modelsDir, '*.mat' ));

imds = imageDatastore(imgsDir,"IncludeSubfolders",true,"LabelSource","foldernames");

AVGtype = "weightAVG";

for i = 1:numel(models)
    load( fullfile( modelsDir, models(i).name ) );
    inputSize = trained_net.Layers(1).InputSize;
    imdsAug = augmentedImageDatastore( inputSize, imds );

    net_name = strrep( models(i).name, 'trained_', '' );
    net_name = strrep( net_name, '.mat', '' );
    net_name = [upper(net_name(1)), net_name(2:end)];

    YPred = classify(trained_net,imdsAug, 'ExecutionEnvironment','cpu', 'Acceleration', 'none');
    %YPred = classify(trained_net,imdsAug);
    
    confusion = confusionmat(imds.Labels,YPred); 
    [microAVG, macroAVG, wAVG, stats] = computeMetrics(confusion);
    
    fprintf( [net_name, ' & '] );
    fprintf([num2str(100*stats{5, AVGtype}, "%.2f") , ' & '])  % Acc
    fprintf([num2str(100*stats{6, AVGtype}, "%.2f") , ' & '])  % Pre
    fprintf([num2str(100*stats{15, AVGtype}, "%.2f") , ' & ']) % Spe
    fprintf([num2str(100*stats{11, AVGtype}, "%.2f") , ' & ']) % Sen
    fprintf([num2str(100*stats{20, AVGtype}, "%.2f") , ' & '])  % F1
    fprintf([num2str(100*stats.macroAVG(23), "%.2f") , ' & ']) % Mavg
    fprintf([num2str(100*stats.macroAVG(24), "%.2f") , ' \\\\ \n']) % Mava
end