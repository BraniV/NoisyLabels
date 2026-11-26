function results = BAMS2gate2(cfg)
%BAMS2GATE2 Levelwise spike-probability rules and a noisy membership teacher.
%   R = BAMS2gate2() runs independent pilot, training and test experiments.
%   R = BAMS2gate2(struct('nRep',10,'doPlot',false)) runs a quick check.
%   Writes risk, teacher-coverage and gate CSVs, plus Times New Roman figures.
%   Base MATLAB only. Gaussian measurement noise allows the ordinary Stein
%   identity. The slab is double exponential (Laplace) at every level.
%   This differs from the Laplace observation model used by original BAMS.
%   Data are generated in an orthonormal wavelet basis. By Parseval, the loss
%   over all coordinates, including untouched scaling coefficients, is
%   signal-domain MSE. An inverse transform is unnecessary for these outputs.
%   cfg fields: seed, nRep, nPilot, nTrain (vector), nTest (signals per repeat),
%   sigma, lambda, useWilson, doPlot, outPrefix. Design constants are below.

if nargin < 1, cfg = struct(); end
defaults = struct('seed',1234,'nRep',200,'nPilot',4,'nTrain',[1 20], ...
    'nTest',12,'sigma',1,'lambda',4,'useWilson',true,'doPlot',true, ...
    'outPrefix','BAMS2gate2');
fields = fieldnames(defaults);
for k = 1:numel(fields)
    name = fields{k};
    if ~isfield(cfg,name), cfg.(name) = defaults.(name); end
end
validateattributes(cfg.nRep,{'numeric'},{'scalar','integer','>=',2});
validateattributes(cfg.nPilot,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.nTrain,{'numeric'},{'vector','integer','positive'});
validateattributes(cfg.nTest,{'numeric'},{'scalar','integer','positive'});
validateattributes(cfg.sigma,{'numeric'},{'scalar','finite','positive'});
validateattributes(cfg.lambda,{'numeric'},{'scalar','finite','nonnegative'});
prefix = char(cfg.outPrefix);
rng(cfg.seed,'twister');

n = 1024;
levels = 3:9;
nLevel = numel(levels);
% Columns denote quiet and active blocks, respectively. Larger j is finer.
piTrue = [linspace(.10,.07,nLevel)', linspace(.38,.28,nLevel)'];
piLo = linspace(.06,.04,nLevel)';
piHi = linspace(.50,.42,nLevel)';
tau = cfg.sigma*linspace(2.8,1.2,nLevel)';
flip = [.04 .16];
bounds = [.05 .20; .025 .10];
caseNames = {'Admissible bounds','Narrow bounds','Labels treated as clean'};
methods = {'Strong endpoint','Weak endpoint','Midpoint','Reference BAMS-type', ...
    'SURE alone','SURE + point teacher','SURE + interval teacher', ...
    'SURE + narrow interval teacher','Context oracle','Bayes oracle'};
nSize = numel(cfg.nTrain);
nMethod = numel(methods);
loss = zeros(cfg.nRep,nSize,nMethod);
contains = false(cfg.nRep,nLevel,2,3);
width = zeros(cfg.nRep,nLevel,2,3);
gateLog = zeros(nLevel,2,nSize,nMethod);
oracle = zeros(nLevel,2);
vEnd = zeros(nLevel,2);

for j = 1:nLevel
    x = linspace(-35,35,10001)'*max(tau(j),cfg.sigma);
    [m,slab,spike] = slabStats(x,cfg.sigma,tau(j));
    A = posterior(m,slab,spike,piLo(j));
    C = posterior(m,slab,spike,piHi(j));
    vEnd(j,:) = trapz(x,(C-A).^2 .* ...
        (exp(slab).*[piLo(j),piHi(j)] + exp(spike).*(1-[piLo(j),piHi(j)])));
    oracle(j,:) = gateScale(piTrue(j,:),piLo(j),piHi(j),vEnd(j,:));
end

for rep = 1:cfg.nRep
    % Eight scaling coefficients are passed through untouched by every rule.
    scalingLoss = cfg.sigma^2*sum(randn(2^levels(1),cfg.nTest).^2,1);
    total = repmat(reshape(scalingLoss,1,1,[]),nSize,nMethod,1);
    for j = 1:nLevel
        count = 2^(levels(j)-1);
        [~,~,H] = draw(count,cfg.nPilot,piTrue(j,:),tau(j),cfg.sigma);
        labels = xor(H,rand(size(H)) < flip(1)+(flip(2)-flip(1))*H);
        q = mean(labels,1);
        [qLo,qHi] = rateLimits(q,size(labels,1),cfg.useWilson);
        teacher = zeros(2,2,3);
        for b = 1:2
            piA = (qLo-bounds(b,1))/(1-bounds(b,1));
            piC = qHi/(1-bounds(b,2));
            teacher(:,:,b) = gateScale([piA(:),piC(:)],piLo(j),piHi(j),vEnd(j,:));
        end
        point = gateScale(q(:),piLo(j),piHi(j),vEnd(j,:));
        teacher(:,:,3) = [point,point];
        for b = 1:3
            contains(rep,j,:,b) = reshape(oracle(j,:)' >= teacher(:,1,b) & ...
                oracle(j,:)' <= teacher(:,2,b),1,1,2);
            width(rep,j,:,b) = reshape(diff(teacher(:,:,b),1,2),1,1,2);
        end

        [D,~] = draw(count,max(cfg.nTrain),piTrue(j,:),tau(j),cfg.sigma);
        [A,C,dB] = actions(D,cfg.sigma,tau(j),piLo(j),piHi(j));
        B = C-A;
        V = B.^2;
        S = B.*(D-A)-cfg.sigma^2*dB;
        [Dt,theta] = draw(count,cfg.nTest,piTrue(j,:),tau(j),cfg.sigma);
        [At,Ct,~,mt,lt,nt] = actions(Dt,cfg.sigma,tau(j),piLo(j),piHi(j));
        ref = posterior(mt,lt,nt,mean(piTrue(j,:)));
        bayes = posterior(mt,lt,nt,piTrue(j,:));

        for k = 1:nSize
            idx = 1:count*cfg.nTrain(k);
            z = sum(S(idx,:),1)./max(sum(V(idx,:),1),realmin);
            gates = [zeros(1,2); ones(1,2); .5*ones(1,2); nan(1,2); ...
                clip(z,0,1); teach(z,teacher(:,:,3),cfg.lambda); ...
                teach(z,teacher(:,:,1),cfg.lambda); ...
                teach(z,teacher(:,:,2),cfg.lambda); oracle(j,:); nan(1,2)];
            gateLog(j,:,k,:) = gateLog(j,:,k,:) + reshape(gates',1,2,1,nMethod);
            for m = 1:nMethod
                if m == 4
                    estimate = ref;
                elseif m == nMethod
                    estimate = bayes;
                else
                    estimate = At+(Ct-At).*gates(m,:);
                end
                err = reshape((estimate-theta).^2,count,cfg.nTest,2);
                total(k,m,:) = total(k,m,:) + reshape(sum(sum(err,1),3),1,1,[]);
            end
        end
        if rep == 1 && j == 4
            exampleTeacher = teacher;
            exampleGates = gates;
        end
    end
    loss(rep,:,:) = reshape(mean(total,3)/n,1,nSize,nMethod);
end

risk = table();
gateTable = table();
for k = 1:nSize
    values = reshape(loss(:,k,:),cfg.nRep,nMethod);
    paired = values-values(:,5);
    risk = [risk; table(repmat(cfg.nTrain(k),nMethod,1),methods(:),mean(values,1)', ...
        std(values,0,1)'/sqrt(cfg.nRep),mean(paired,1)',std(paired,0,1)'/sqrt(cfg.nRep), ...
        'VariableNames',{'trainingSignals','method','AMSE','MCSE','diffVsSURE','pairedMCSE'})]; %#ok<AGROW>
    for m = [5 6 7 8 9]
        avg = reshape(gateLog(:,:,k,m),nLevel,2)/cfg.nRep;
        gateTable = [gateTable; table(repmat(cfg.nTrain(k),nLevel,1),levels(:), ...
            repmat(methods(m),nLevel,1),avg(:,1),avg(:,2), ...
            'VariableNames',{'trainingSignals','level','method','quietGate','activeGate'})]; %#ok<AGROW>
    end
end
coverage = table();
for b = 1:3
    cellCoverage = reshape(mean(contains(:,:,:,b),1),nLevel,2);
    cellWidth = reshape(mean(width(:,:,:,b),1),nLevel,2);
    coverage = [coverage; table(levels(:),repmat(caseNames(b),nLevel,1), ...
        cellCoverage(:,1),cellCoverage(:,2),cellWidth(:,1),cellWidth(:,2), ...
        'VariableNames',{'level','teacher','quietCoverage','activeCoverage','quietWidth','activeWidth'})]; %#ok<AGROW>
end
results = struct('risk',risk,'coverage',coverage,'gates',gateTable,'loss',loss,'cfg',cfg);
results.design = table(levels(:),tau,1-piLo,1-piHi,piTrue(:,1),piTrue(:,2), ...
    'VariableNames',{'level','tau','epsStrong','epsWeak','piQuiet','piActive'});
disp(results.design);
disp(risk);
disp(coverage);
if ~isempty(prefix)
    writetable(results.design,[prefix '_design.csv']);
    writetable(risk,[prefix '_risk.csv']);
    writetable(coverage,[prefix '_coverage.csv']);
    writetable(gateTable,[prefix '_gates.csv']);
end
if cfg.doPlot
    makePlots(prefix,cfg,levels(4),tau(4),piLo(4),piHi(4), ...
        exampleTeacher,exampleGates,methods,loss);
end
end

function [D,theta,H] = draw(count,nSignals,pi,tau,sigma)
% Two contiguous blocks of equal size at each level, independently replicated.
H = rand(count*nSignals,2) < pi;
theta = tau*(log(rand(size(H)))-log(rand(size(H)))).*H;
D = theta+sigma*randn(size(H));
end

function [A,C,dB,m,slab,spike] = actions(d,sigma,tau,piLo,piHi)
[m,slab,spike,dm] = slabStats(d,sigma,tau);
[A,wA] = posterior(m,slab,spike,piLo);
[C,wC] = posterior(m,slab,spike,piHi);
dB = (wC-wA).*dm + (wC.*(1-wC)-wA.*(1-wA)).*(m/sigma).^2;
end

function [m,slab,spike,dm] = slabStats(d,sigma,tau)
% Normal-Laplace marginal. Log-CDF evaluation avoids tail underflow.
zp = d/sigma-sigma/tau;
zn = -d/sigma-sigma/tau;
lp = logPhi(zp);
ln = logPhi(zn);
plus = -d/tau+lp;
minus = d/tau+ln;
top = max(plus,minus);
sumLog = top+log(exp(plus-top)+exp(minus-top));
r = exp(minus-sumLog);
m = d+(sigma^2/tau)*(2*r-1);
slab = sigma^2/(2*tau^2)-log(2*tau)+sumLog;
spike = -.5*(d/sigma).^2-log(sigma*sqrt(2*pi));
if nargout > 3
    mills = @(z,l) exp(-.5*z.^2-.5*log(2*pi)-l);
    dm = 1+(2*sigma^2/tau)*r.*(1-r).* ...
        (2/tau-(mills(zp,lp)+mills(zn,ln))/sigma);
end
end

function y = logPhi(z)
y = log(.5*erfc(-z/sqrt(2)));
neg = z < 0;
y(neg) = log(.5*erfcx(-z(neg)/sqrt(2)))-.5*z(neg).^2;
end

function [delta,w] = posterior(m,slab,spike,inclusion)
% P(slab|d) and posterior mean.
w = 1./(1+exp(spike-slab+log1p(-inclusion)-log(inclusion)));
delta = w.*m;
end

function p = gateScale(pi,lo,hi,V)
t = clip((pi-lo)/(hi-lo),0,1);
p = t*V(2)./((1-t)*V(1)+t*V(2));
end

function p = teach(z,interval,lambda)
% Exact minimizer of (p-z)^2+lambda*dist(p,interval)^2 on [0,1].
projection = clip(z,interval(:,1)',interval(:,2)');
p = clip(projection+(z-projection)/(1+lambda),0,1);
end

function y = clip(x,lo,hi)
y = min(max(x,lo),hi);
end

function [lo,hi] = rateLimits(q,n,useWilson)
if useWilson
    z = 1.95996398454005;
    denominator = 1+z^2/n;
    middle = (q+z^2/(2*n))/denominator;
    half = z*sqrt(q.*(1-q)/n+z^2/(4*n^2))/denominator;
    lo = clip(middle-half,0,1);
    hi = clip(middle+half,0,1);
else
    lo = q;
    hi = q;
end
end

function makePlots(prefix,cfg,j,tau,piLo,piHi,teacher,gates,methods,loss)
font = 'Times New Roman';
f = figure('Color','w','Units','inches','Position',[1 1 8.5 3.7]);
ax = subplot(1,2,1); hold(ax,'on');
x = linspace(-6,6,601)'*cfg.sigma;
[A,C] = actions(x,cfg.sigma,tau,piLo,piHi);
s = cfg.sigma;
fill(ax,[x;flipud(x)]/s,[A;flipud(C)]/s,[.87 .91 .95],'EdgeColor','none', ...
    'HandleVisibility','off');
plot(ax,x/s,x/s,':','Color',[.5 .5 .5],'HandleVisibility','off');
plot(ax,x/s,A/s,'-','Color',[.12 .30 .53],'LineWidth',1.7);
plot(ax,x/s,C/s,'--','Color',[.61 .23 .12],'LineWidth',1.7);
xlabel(ax,'d / \sigma'); ylabel(ax,'\delta(d) / \sigma');
title(ax,sprintf('Level %d, DE scale %.2f',j,tau),'FontWeight','normal');
legend(ax,{sprintf('Strong, \\epsilon = %.2f',1-piLo),sprintf('Weak, \\epsilon = %.2f',1-piHi)}, ...
    'Location','northwest','Box','off');
box(ax,'on');

ax = subplot(1,2,2); hold(ax,'on');
for c = 1:2
    line(ax,teacher(c,:,1),[c c],'Color',[.73 .82 .91],'LineWidth',9, ...
        'HandleVisibility','off');
end
h1 = plot(ax,gates(5,:),[1 2]-.10,'ks','MarkerSize',6);
h2 = plot(ax,gates(7,:),[1 2],'o','Color',[.12 .30 .53],'MarkerFaceColor',[.12 .30 .53]);
h3 = plot(ax,gates(9,:),[1 2]+.10,'kd','MarkerFaceColor','k','MarkerSize',6);
set(ax,'YTick',[1 2],'YTickLabel',{'Quiet','Active'},'YLim',[.62 2.38],'XLim',[-.03 1.03]);
xlabel(ax,'Gate weight on weak rule');
title(ax,sprintf('Interval teacher, %d training signals',cfg.nTrain(end)), ...
    'FontWeight','normal');
legend(ax,[h1 h2 h3],{'SURE','SURE + interval','Context oracle'},'Location','southoutside','Box','off');
box(ax,'on');
formatFigure(f,font,[prefix '_rules_teacher'],prefix);

f = figure('Color','w','Units','inches','Position',[1 1 7.5 4.2]);
ax = axes(f); hold(ax,'on');
idx = [6 7 8 9 10];
row = 1:numel(idx);
for k = 1:numel(cfg.nTrain)
    diffRisk = reshape(loss(:,k,idx)-loss(:,k,5),cfg.nRep,[]);
    offset = .55*(k-(numel(cfg.nTrain)+1)/2)/numel(cfg.nTrain);
    errorbar(ax,mean(diffRisk),row+offset,std(diffRisk,0,1)/sqrt(cfg.nRep), ...
        'horizontal','o','LineStyle','none','DisplayName', ...
        sprintf('%d training signal(s)',cfg.nTrain(k)));
end
xline(ax,0,':','HandleVisibility','off');
set(ax,'YTick',row,'YTickLabel',methods(idx),'YDir','reverse','YLim',[.5 numel(row)+.5]);
xlabel(ax,'AMSE difference from SURE alone');
legend(ax,'Location','southoutside','Box','off'); box(ax,'on');
formatFigure(f,font,[prefix '_risk'],prefix);
end

function formatFigure(f,font,name,prefix)
set(findall(f,'-property','FontName'),'FontName',font);
set(findall(f,'-property','FontSize'),'FontSize',11);
if isempty(prefix), return; end
if exist('exportgraphics','file')
    exportgraphics(f,[name '.pdf'],'ContentType','vector');
    exportgraphics(f,[name '.png'],'Resolution',400);
else
    set(f,'PaperPositionMode','auto');
    print(f,name,'-dpdf','-painters');
    print(f,name,'-dpng','-r400');
end
end
