function results = BAMS2gate1(cfg)
%BAMS2GATE1 Compare two pairs of levelwise DE slab scales on the same data.
%   R = BAMS2gate1() runs the Doppler experiment and writes CSV, PDF and PNG.
%   R = BAMS2gate1(struct('signal','bumps')) repeats the design on Bumps.
%   cfg.doPlot=false skips figures. cfg.outPrefix sets the output prefix.
%   Both pairs use identical training and test draws with separate fitted gates.
%   The original BAMS calibration and gate objectives are retained. SURE uses
%   derivatives with BAMS nuisance estimates held fixed and is approximate here.

if nargin < 1, cfg = struct(); end
if ~isfield(cfg,'outPrefix'), cfg.outPrefix = 'BAMS2gate1'; end
prefix = cfg.outPrefix;
cfg.family = 'tau';
cfg.mult = [0.6 1.6];
if ~isempty(prefix), cfg.outPrefix = [prefix '_tau06_16']; end
results.standard = runPair(cfg);
cfg.mult = [0.3 3.0];
if ~isempty(prefix), cfg.outPrefix = [prefix '_tau03_30']; end
results.wide = runPair(cfg);

nTest = size(results.standard.AMSE,1);
change = results.wide.AMSE - results.standard.AMSE;
am = [results.standard.table.AMSE, results.wide.table.AMSE];
best = min(am(2:3,:),[],1);
results.comparison = table(results.standard.table.method, am(:,1), am(:,2), ...
    am(:,1)-best(1), am(:,2)-best(2), mean(change,1)', std(change,0,1)'/sqrt(nTest), ...
    'VariableNames', {'method','AMSE_standard','AMSE_wide','vsBestEndpoint_standard', ...
    'vsBestEndpoint_wide','wideMinusStandard','pairedMCSE'});
disp(results.comparison);
if ~isempty(prefix), writetable(results.comparison,[prefix '_comparison.csv']); end
end

function results = runPair(cfg)
%BAMS2GATE  Two levelwise BAMS rules and a trained gate between them.
%
%   results = BAMS2gate(cfg)
%
%   At each detail level j, BAMS (Vidakovic and Ruggeri, 2001) is run twice. With
%   family = 'tau' (default) the double-exponential slab scale is mult(1)*tau_j and
%   mult(2)*tau_j, where tau_j is the standard BAMS scale, and mu (noise) and eps_j (spike
%   weight) are held fixed. With family = 'eps' the slab is kept and the prior odds of the
%   spike, eps_j/(1 - eps_j), are multiplied by mult(1) and mult(2) instead. For each
%   coefficient the output of smaller magnitude is the strong rule A and the larger is
%   the weak rule C. A gate p_j(X) in [0,1] interpolates, delta = A + (C - A) p_j(X).
%
%   The gate is trained in four ways, all on independent training realizations:
%     'pair'  squared error against an independent replicate Y = theta + noise,
%             i.e. minimize sum (Y - A - B p)^2, with B = C - A;
%     'sure'  Stein's unbiased risk estimate from the noisy signal alone, no replicate:
%             minimize sum (V p^2 - 2 S p), V = B^2, S = B (D - A) - sigma^2 dB/dD
%             (context-only features, so that dp/dD = 0);
%     'vote'  logistic classification of the replicate's vote
%             1{(Y - C)^2 < (Y - A)^2} (a noisy label of the winning rule);
%     'wvote' the same vote, weighted by its cost |(Y - A)^2 - (Y - C)^2|.
%   'sureself' fits the SURE gate on the test signal itself (single-signal use).
%   Both squared-error objectives have the common form sum (V p^2 - 2 S p).
%
%   Features per coefficient (from the first noisy signal, divided by sigma^2, log1p):
%     mag = d^2;  nb = mean of the two same-level neighbours squared;
%     pc  = parent squared and mean of the two children squared (0 where absent).
%
%   cfg fields (defaults in brackets): signal ['doppler'] ('doppler','bumps','blocks',
%   'heavisine'), n [1024], rsnr [5], sigma [1], family ['tau'], mult [0.6 1.6],
%   Rtr [20], Rte [100],
%   seed [20261125], filt [Symmlet 4], L [log2(n)-3], Cscale [2], gam [2],
%   ridge [1e-3], ridgeSelf [1e-2], plotLevel [log2(n)-3], doPlot [true],
%   outPrefix ['BAMS2gate'].
%
%   Needs base MATLAB only. Wavmat and the BAMS rule follow BAMS.m (Vidakovic).

if nargin < 1, cfg = struct(); end
def = struct('signal','doppler','n',1024,'rsnr',5,'sigma',1,'family','tau','mult',[0.6 1.6], ...
    'Rtr',20,'Rte',100,'seed',20261125,'filt',[],'L',[],'Cscale',2,'gam',2, ...
    'ridge',1e-3,'ridgeSelf',1e-2,'plotLevel',[],'doPlot',true,'outPrefix','BAMS2gate');
fn = fieldnames(def);
for k = 1:numel(fn), if ~isfield(cfg, fn{k}), cfg.(fn{k}) = def.(fn{k}); end, end
n = cfg.n; J = log2(n);
if isempty(cfg.filt)
    cfg.filt = [-0.07576571478934 -0.02963552764595 0.49761866763246 0.80373875180522 ...
                 0.29785779560554 -0.09921954357694 -0.01260396726226 0.03222310060407];
end
if isempty(cfg.L), cfg.L = J - 3; end
if isempty(cfg.plotLevel), cfg.plotLevel = J - 3; end
coarsest = J - cfg.L; LEV = coarsest:(J-1); sig = cfg.sigma;
rng(cfg.seed, 'twister');

W = Wavmat(cfg.filt, n, cfg.L, 2);
f0 = testsignal(cfg.signal, n); f0 = f0/std(f0)*cfg.rsnr*sig;
theta = W*f0(:);

modes = {'const', {}; 'mag', {'mag'}; 'mag+nb', {'mag','nb'}; ...
         'mag+nb+pc', {'mag','nb','pc'}; 'ctx', {'nb','pc'}};

% ------------------------------------------------------------------ training data
TR = cell(J,1);
for r = 1:cfg.Rtr
    D = theta + sig*randn(n,1); Y = theta + sig*randn(n,1);
    [mu, ep, ta] = bamspar(D, n, J, coarsest, cfg);
    [A, C, dB] = endpoints(D, mu, ep, ta, LEV, cfg.mult, cfg.family);
    for j = LEV
        k = lind(j); B = C(k) - A(k);
        rec.B = B; rec.Spair = B.*(Y(k) - A(k));
        rec.Ssure = B.*(D(k) - A(k)) - sig^2*dB(k);
        rec.vote = double((Y(k) - C(k)).^2 < (Y(k) - A(k)).^2);
        rec.ovote = double((theta(k) - C(k)).^2 < (theta(k) - A(k)).^2);
        rec.rt = (Y(k) - A(k)).^2 - (Y(k) - C(k)).^2;
        rec.absd = abs(D(k))/sig;
        for m = 1:size(modes,1), rec.(fld(modes{m,1})) = features(D, j, modes{m,2}, coarsest, J, sig); end
        TR{j} = appendrec(TR{j}, rec);
    end
end

% ------------------------------------------------------------------ fit gates
G = struct();
for j = LEV
    T = TR{j}; V = T.B.^2;
    for m = 1:size(modes,1)
        G.(sprintf('pair_%s_%d', fld(modes{m,1}), j)) = fitgate(T.(fld(modes{m,1})), T.Spair, V, cfg.ridge);
    end
    G.(sprintf('sure_ctx_%d', j))   = fitgate(T.(fld('ctx')),   T.Ssure, V, cfg.ridge);
    G.(sprintf('sure_const_%d', j)) = fitgate(T.(fld('const')), T.Ssure, V, cfg.ridge);
    X = T.(fld('mag+nb+pc'));
    G.(sprintf('vote_%d', j))  = fitlogit(X, T.vote, ones(size(V)), cfg.ridge);
    G.(sprintf('wvote_%d', j)) = fitlogit(X, double(T.rt > 0), abs(T.rt), cfg.ridge);
end
allv = []; allo = []; alld = [];
for j = LEV, allv = [allv; TR{j}.vote]; allo = [allo; TR{j}.ovote]; alld = [alld; TR{j}.absd]; end %#ok<AGROW>
results.voteAgreement = mean(allv == allo);

% ------------------------------------------------------------------ test
names = {'BAMS (reference)', sprintf('BAMS, %s x %.2g', cfg.family, cfg.mult(1)), sprintf('BAMS, %s x %.2g', cfg.family, cfg.mult(2)), ...
         'Midpoint', 'Pair: constant', 'Pair: magnitude', 'Pair: mag+neighbours', ...
         'Pair: mag+nb+parent/children', 'Pair: context only', 'SURE: constant', ...
         'SURE: context only', 'SURE on test signal', 'Votes (unweighted)', 'Votes (cost-weighted)', ...
         'Oracle coefficientwise'};
AM = zeros(cfg.Rte, numel(names));
for r = 1:cfg.Rte
    D = theta + sig*randn(n,1);
    [mu, ep, ta] = bamspar(D, n, J, coarsest, cfg);
    [A, C, dB, R0, E1, E2] = endpoints(D, mu, ep, ta, LEV, cfg.mult, cfg.family);
    out = cell(1, numel(names));
    out{1} = R0; out{2} = E1; out{3} = E2;
    out{4} = gated(D, A, C, LEV, @(j) 0.5);
    pm = {'const','mag','mag+nb','mag+nb+pc','ctx'};
    for m = 1:5
        md = modes{strcmp(modes(:,1), pm{m}), 2};
        out{4+m} = gated(D, A, C, LEV, @(j) logist(features(D, j, md, coarsest, J, sig)* ...
                         G.(sprintf('pair_%s_%d', fld(pm{m}), j))));
    end
    out{10} = gated(D, A, C, LEV, @(j) logist(features(D, j, {}, coarsest, J, sig)*G.(sprintf('sure_const_%d', j))));
    out{11} = gated(D, A, C, LEV, @(j) logist(features(D, j, {'nb','pc'}, coarsest, J, sig)*G.(sprintf('sure_ctx_%d', j))));
    out{12} = gated(D, A, C, LEV, @(j) selfsure(D, A, C, dB, j, coarsest, J, sig, cfg.ridgeSelf));
    out{13} = gated(D, A, C, LEV, @(j) logist(features(D, j, {'mag','nb','pc'}, coarsest, J, sig)*G.(sprintf('vote_%d', j))));
    out{14} = gated(D, A, C, LEV, @(j) logist(features(D, j, {'mag','nb','pc'}, coarsest, J, sig)*G.(sprintf('wvote_%d', j))));
    out{15} = gated(D, A, C, LEV, @(j) oraclep(theta, A, C, j));
    for m = 1:numel(names), AM(r, m) = mean((out{m} - theta).^2); end   % = signal-domain MSE (W orthonormal)
    if r == 1, keepD = D; keepOut = out; keepPar = {mu, ep, ta, A, C}; end
end
dif = AM - AM(:,1);
results.table = table(names(:), mean(AM,1)', std(AM,0,1)'/sqrt(cfg.Rte), mean(dif,1)', ...
    std(dif,0,1)'/sqrt(cfg.Rte), 'VariableNames', {'method','AMSE','MCSE','diffVsBAMS','MCSEdiff'});
results.AMSE = AM; results.cfg = cfg; results.gates = G;
fprintf('\n%s, n = %d, RSNR = %g, sigma = %g, %s multipliers [%g %g]\n', cfg.signal, n, cfg.rsnr, sig, cfg.family, cfg.mult);
fprintf('Replicate votes agree with the clean-signal vote for %.1f%% of training coefficients\n\n', 100*results.voteAgreement);
disp(results.table);
if ~isempty(cfg.outPrefix), writetable(results.table, [cfg.outPrefix '_results.csv']); end

% ------------------------------------------------------------------ figures
if cfg.doPlot
    oldAxesFont = get(groot,'defaultAxesFontName');
    oldTextFont = get(groot,'defaultTextFontName');
    oldFontSize = get(groot,'defaultAxesFontSize');
    restoreFonts = onCleanup(@() set(groot,'defaultAxesFontName',oldAxesFont, ...
        'defaultTextFontName',oldTextFont,'defaultAxesFontSize',oldFontSize)); %#ok<NASGU>
    set(groot, 'defaultAxesFontName', 'Times New Roman', 'defaultTextFontName', 'Times New Roman', 'defaultAxesFontSize', 11);
    j = cfg.plotLevel; mu = keepPar{1}; ep = keepPar{2}; ta = keepPar{3};
    x = linspace(-8, 8, 801)'*sig;
    [t1, e1, t2, e2] = rulepars(ta(j), ep(j), cfg.mult, cfg.family);
    r1 = bamsrule(x, mu, t1, e1); r2 = bamsrule(x, mu, t2, e2);
    r0 = bamsrule(x, mu, ta(j), ep(j));
    lo = sign(x).*min(abs(r1), abs(r2)); hi = sign(x).*max(abs(r1), abs(r2));
    f1 = figure('Color','w','Units','inches','Position',[1 1 6.5 3]);
    subplot(1,2,1); hold on
    fill([x; flipud(x)]/sig, [lo; flipud(hi)]/sig, [0.85 0.85 0.85], 'EdgeColor', 'none');
    plot(x/sig, r1/sig, 'k--', x/sig, r2/sig, 'k:', x/sig, r0/sig, 'k-', 'LineWidth', 1.2);
    plot(x/sig, x/sig, '-', 'Color', [0.6 0.6 0.6]);
    xlabel('d/\sigma'); ylabel('\delta(d)/\sigma'); box on; axis([-8 8 -8 8]);
    legend({'between rules', sprintf('\\tau \\times %.2g', cfg.mult(1)), ...
            sprintf('\\tau \\times %.2g', cfg.mult(2)), 'BAMS'}, 'Location', 'northwest', 'Box', 'off');
    subplot(1,2,2); hold on
    k = lind(j); Fx = features(keepD, j, {'nb','pc'}, coarsest, J, sig);
    pg = logist(Fx*G.(sprintf('sure_ctx_%d', j)));
    scatter(Fx(:,2), pg, 14, abs(keepD(k))/sig, 'filled'); colormap(gca, parula);
    cb = colorbar; cb.Label.String = '|d|/\sigma'; ylim([0 1]);
    xlabel('log(1 + neighbour energy)'); ylabel('gate p_j (0 = strong, 1 = weak)'); box on
    if ~isempty(cfg.outPrefix), exportfig(f1, [cfg.outPrefix '_rules']); end

    f2 = figure('Color','w','Units','inches','Position',[1 1 6.5 2.8]);
    edges = 0:0.5:6; mid = edges(1:end-1) + 0.25; fv = nan(size(mid)); fo = fv; ag = fv;
    for b = 1:numel(mid)
        s = alld >= edges(b) & alld < edges(b+1);
        if nnz(s) > 20, fv(b) = mean(allv(s)); fo(b) = mean(allo(s)); ag(b) = mean(allv(s) == allo(s)); end
    end
    subplot(1,2,1); plot(mid, fv, 'ko-', mid, fo, 'ks--', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
    xlabel('|d|/\sigma'); ylabel('fraction voting for the weak rule'); ylim([0 1]); box on
    legend({'replicate votes', 'clean-signal votes'}, 'Box', 'off', 'Location', 'best');
    subplot(1,2,2); plot(mid, ag, 'ko-', 'MarkerFaceColor', 'k', 'MarkerSize', 4); yline(0.5, ':');
    xlabel('|d|/\sigma'); ylabel('agreement with clean vote'); ylim([0.4 1]); box on
    if ~isempty(cfg.outPrefix), exportfig(f2, [cfg.outPrefix '_votes']); end

    f3 = figure('Color','w','Units','inches','Position',[1 1 6.5 3.6]);
    md = mean(dif,1); se = std(dif,0,1)/sqrt(cfg.Rte); y = numel(names):-1:1;
    errorbar(md(2:end), y(2:end), se(2:end), 'horizontal', 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
    xline(0, ':'); set(gca, 'YTick', fliplr(y(2:end)), 'YTickLabel', fliplr(names(2:end)), 'YLim', [0.5 numel(names)-0.5]);
    xlabel('AMSE difference from BAMS (negative = better)'); box on
    if ~isempty(cfg.outPrefix), exportfig(f3, [cfg.outPrefix '_amse']); end
end
end

% ===================================================================================== helpers
function s = fld(name), s = ['X_' strrep(strrep(strrep(name, '+', '_'), '/', '_'), ' ', '_')]; end
function k = lind(j), k = (2^j + 1):2^(j+1); end
function p = logist(z), p = 1./(1 + exp(-z)); end

function S = appendrec(S, rec)
f = fieldnames(rec);
if isempty(S), S = rec; return; end
for i = 1:numel(f), S.(f{i}) = [S.(f{i}); rec.(f{i})]; end
end

function f = testsignal(name, n)
t = (1:n)'/n;
switch lower(name)
    case 'doppler', f = sqrt(t.*(1-t)).*sin(2.1*pi./(t + 0.05));
    case 'heavisine', f = 4*sin(4*pi*t) - sign(t - 0.3) - sign(0.72 - t);
    case 'blocks'
        tj = [.1 .13 .15 .23 .25 .4 .44 .65 .76 .78 .81]; hj = [4 -5 3 -4 5 -4.2 2.1 4.3 -3.1 2.1 -4.2];
        f = zeros(n,1); for k = 1:11, f = f + hj(k)*(1 + sign(t - tj(k)))/2; end
    case 'bumps'
        tj = [.1 .13 .15 .23 .25 .4 .44 .65 .76 .78 .81]; hj = [4 5 3 4 5 4.2 2.1 4.3 3.1 5.1 4.2];
        wj = [.005 .005 .006 .01 .01 .03 .01 .01 .005 .008 .005];
        f = zeros(n,1); for k = 1:11, f = f + hj(k)./(1 + abs((t - tj(k))/wj(k))).^4; end
    otherwise, error('BAMS2gate:signal', 'Unknown signal %s.', name);
end
end

function [mu, ep, ta] = bamspar(sw, n, J, coarsest, cfg)
% BAMS parameters exactly as in BAMS.m: mu from the finest-level IQR, eps_j and tau_j by level.
niz = sort(sw((2^(J-1)+1):2^J)); mm = numel(niz);
lo = floor(1/4*mm); hi = floor(3/4*mm);
pseudos = abs(niz(hi) - niz(lo))/cfg.Cscale; mu = 1/pseudos^2;
tauu = sqrt(max(std(sw)^2 - 1/mu, 0.0001));
ep = zeros(J,1); ta = zeros(J,1);
for i = (J-1):-1:coarsest
    ep(i) = 1 - 1/(i - coarsest + 1)^cfg.gam; ta(i) = tauu*sqrt(i);
end
end

function br = bamsrule(d, mu, tau, eps)
% Vectorized BAMS Bayes rule (same values as bayesrule in BAMS.m), written in a form that
% does not overflow or underflow for large |d| when sqrt(2 mu) > 1/tau.
a = abs(d); s2 = sqrt(2*mu); k = tau^2 - 1/(2*mu);
if abs(k) < 1e-8, tau = tau*(1 + 1e-6); k = tau^2 - 1/(2*mu); end
x = exp(-(s2 - 1/tau)*a);
b = (tau*k*a + tau^2/mu*(x - 1))./(k*(tau - x/s2));
R = (tau - x/s2)./((2*tau^2 - 1/mu)*0.5*s2).*exp((s2 - 1/tau)*a);   % slab/spike marginal ratio
w = 1./(1 + eps./((1 - eps)*R));
br = sign(d).*w.*b;
end

function [t1, e1, t2, e2] = rulepars(tau, ep, mult, family)
% parameters of the two candidate rules at one level
if strcmpi(family, 'eps')
    o = ep/(1 - ep); t1 = tau; t2 = tau;
    e1 = o*mult(1)/(1 + o*mult(1)); e2 = o*mult(2)/(1 + o*mult(2));
else
    t1 = mult(1)*tau; t2 = mult(2)*tau; e1 = ep; e2 = ep;
end
end

function [A, C, dB, R0, E1, E2] = endpoints(sw, mu, ep, ta, LEV, mult, family)
% A = output of smaller magnitude, C = larger, per coefficient; dB = d(C - A)/dD numerically.
A = sw; C = sw; dB = zeros(size(sw)); R0 = sw; E1 = sw; E2 = sw; h = 1e-4;
for j = LEV
    k = lind(j); d = sw(k);
    [t1, e1, t2, e2] = rulepars(ta(j), ep(j), mult, family);
    r1 = @(z) bamsrule(z, mu, t1, e1); r2 = @(z) bamsrule(z, mu, t2, e2);
    [a, c] = ordered(r1(d), r2(d)); [ap, cp] = ordered(r1(d + h), r2(d + h)); [am, cm] = ordered(r1(d - h), r2(d - h));
    A(k) = a; C(k) = c; dB(k) = ((cp - ap) - (cm - am))/(2*h);
    R0(k) = bamsrule(d, mu, ta(j), ep(j)); E1(k) = r1(d); E2(k) = r2(d);
end
end

function [a, c] = ordered(u, v)
s = abs(u) <= abs(v); a = u; a(~s) = v(~s); c = v; c(~s) = u(~s);
end

function X = features(sw, j, md, coarsest, J, sig)
d = sw(lind(j)); nn = numel(d); X = ones(nn, 1);
if any(strcmp(md, 'mag')), X = [X, log1p(d.^2/sig^2)]; end
if any(strcmp(md, 'nb')),  X = [X, log1p(0.5*(circshift(d,1).^2 + circshift(d,-1).^2)/sig^2)]; end
if any(strcmp(md, 'pc'))
    if j - 1 >= coarsest, par = sw(lind(j-1)); par = par(floor((0:nn-1)'/2) + 1); else, par = zeros(nn,1); end
    if j + 1 <= J - 1, ch = sw(lind(j+1)); chl = 0.5*(ch(1:2:end).^2 + ch(2:2:end).^2); else, chl = zeros(nn,1); end
    X = [X, log1p(par.^2/sig^2), log1p(chl/sig^2)];
end
end

function z = gated(D, A, C, LEV, pfun)
z = A;
for j = LEV, k = lind(j); z(k) = A(k) + (C(k) - A(k)).*pfun(j); end
end

function p = oraclep(theta, A, C, j)
k = lind(j); B = C(k) - A(k); p = 0.5*ones(size(B)); s = abs(B) > 1e-12;
p(s) = min(max((theta(k(s)) - A(k(s)))./B(s), 0), 1);
end

function p = selfsure(D, A, C, dB, j, coarsest, J, sig, ridge)
k = lind(j); B = C(k) - A(k); X = features(D, j, {'nb','pc'}, coarsest, J, sig);
b = fitgate(X, B.*(D(k) - A(k)) - sig^2*dB(k), B.^2, ridge); p = logist(X*b);
end

function b = fitgate(X, S, V, ridge)
% minimize sum(V p^2 - 2 S p) + ridge |b|^2 over b, p = logistic(X b); Gauss-Newton with step halving.
b = zeros(size(X,2),1); Lb = gloss(b);
for it = 1:50
    p = logist(X*b); dp = p.*(1 - p);
    g = X'*((2*V.*p - 2*S).*dp) + 2*ridge*b;
    H = X'*(X.*(2*V.*dp.^2)) + (2*ridge + 1e-8)*eye(numel(b));
    st = H\g; a = 1;
    while a > 1e-4
        nb = b - a*st; Ln = gloss(nb);
        if Ln <= Lb, b = nb; Lb = Ln; break; end
        a = a/2;
    end
    if norm(a*st) < 1e-7, break; end
end
    function L = gloss(bb), pp = logist(X*bb); L = sum(V.*pp.^2 - 2*S.*pp) + ridge*(bb'*bb); end
end

function b = fitlogit(X, y, w, ridge)
% weighted logistic regression by Newton-Raphson
b = zeros(size(X,2),1);
for it = 1:50
    p = logist(X*b); g = X'*(w.*(p - y)) + ridge*b;
    H = X'*(X.*(w.*p.*(1 - p))) + ridge*eye(numel(b));
    st = H\g; b = b - st;
    if norm(st) < 1e-8, break; end
end
end

function exportfig(f, name)
if exist('exportgraphics', 'file')
    exportgraphics(f, [name '.pdf'], 'ContentType', 'vector');
    exportgraphics(f, [name '.png'], 'Resolution', 400);
else
    set(f, 'PaperPositionMode', 'auto'); print(f, name, '-dpdf', '-painters'); print(f, name, '-dpng', '-r400');
end
end

function W = Wavmat(h, N, k0, shift)
% Wavmat -- matrix of the periodic orthogonal DWT (Vidakovic 1999, pp. 115-116); as in BAMS.m.
if nargin == 3, shift = 2; end
J = log2(N);
if nargin == 2, shift = 2; k0 = J; end
h = h(:)'; g = fliplr(conj(h).*(-1).^(1:length(h)));
if J ~= floor(J), error('For standard DWT, N has to be a power of 2.'); end
h = [h, zeros(1,N)]; g = [g, zeros(1,N)];
oldmat = eye(2^(J-k0));
for k = k0:-1:1
    ubJk = 2^(J-k); ubJk1 = 2^(J-k+1);
    hmat = zeros(ubJk1, ubJk); gmat = hmat;
    for jj = 1:ubJk
        for ii = 1:ubJk1
            modulus = mod(N + ii - 2*jj + shift, ubJk1);
            modulus = modulus + (modulus == 0)*ubJk1;
            hmat(ii,jj) = h(modulus); gmat(ii,jj) = g(modulus);
        end
    end
    W = [oldmat*hmat'; gmat'];
    oldmat = W;
end
end
