function A = EnKF_l63(E)
%% EnKF_l63.m
%
% Run the Lorenz 1963 model and 
% assimilate observations using the Ensemble Kalman Filter

% Lisa Neef; started 17 June 2013
%
% INPUT:
%   E: a Matlab structure that holds all the model and assimilation
%	parameters. It is generated using E = set_enkf_inputs
%
% OUTPUT:
%   A: a Matlab structure that holds the truth,observed, and analysis values 
% 	for the model variables x,y,and z 
%		A.xt
%		A.yt
%		A.zt
%		A.xo
%		A.yo
%		A.zo
%		A.xa
%		A.ya
%		A.za
%	as well as the average state error (i.e. the true error) 
%		A.ETave
%	and the average analysis (or estimated) error
%		A.EAave
%-----------------------------------------------------------


%% extract the individual assimilation parameters from E
sigma 	= E.sigma;
rho 	= E.rho;
beta 	= E.beta;

dt      = E.dt;
Tend 	= E.Tend;
N       = E.N;
tobs 	= E.tobs;

xt0 	= E.xt0;
xf0 	= E.xf0;
sig0 	= E.sig0;           % initial forecast error variance
sig_obs = E.sig_obs;        % observation error covariance

obsx	= E.obsx;
obsy	= E.obsy;
obsz	= E.obsz;

obs_meanxy	= E.obs_meanxy;
obs_meanyz	= E.obs_meanyz;
obs_meanxz	= E.obs_meanxz;

%% Initialize arrays to hold everything
t       = 1:dt:Tend;
nT      = length(t);
XT      = zeros(3,nT)+NaN;	% array to hold the true state in time
XENS 	= zeros(N,3,nT)+NaN;% array to hold the ensembles' states in time
S       = zeros(3,nT)+NaN;	% array to hold analysis error variance in time
YOBS 	= zeros(3,nT)+NaN;

%% initial conditions
XT(:,1) = xt0;			% initial truth
% xf 	= xt0+sig0*rand(3,1); % initial forecast
% [v] xf0 init already in 'E' ('randn' normal dist, sig0 std dev from xt0)
for iens = 1:N %[vasu] for each ensemble iter, w/ total N
  XENS(iens,:,1) = xf0 + sig0*randn(3,1); %[v] init 'prior' as normal dist
  % with mean xf0 and std sig0; init with diff rand state for each ensemble
end
%[v] all init forecast states for the ensemble= XENS(:,:,1)

%% define the observation error covariance matrix  
R = sig_obs*eye(3);

%% define the observation operator, based on the desired obs configuration 

% note that this code is not (yet) equipped to handle cases were we observe
% - single obs AND averages -- throw an error in this case
observe_averages = obs_meanxy+obs_meanyz+obs_meanxz;
observe_single_variables = obsx+obsy+obsz;
if (observe_averages > 0) && (observe_single_variables > 0)
	error('Cannot observe both variable averages and single variables...please change the input file.')
end

H = zeros(3); %[v] obs operator
if obsx, H(1,1) = 1; end
if obsy, H(2,2) = 1; end
if obsz, H(3,3) = 1; end

if observe_averages > 0
	if obs_meanxy
		H(1,1) = 0.5;
		H(1,2) = 0.5;
	end
	if obs_meanyz
		H(2,2) = 0.5;
		H(2,3) = 0.5;
	end
	if obs_meanxz
		H(3,1) = 0.5;
		H(3,3) = 0.5;
	end
end

%% Loop in time
xfens = squeeze(XENS(:,:,1));
for k = 1:nT-1 %[vasu] for each time

	if E.run_filter	%[vasu] optional run Kalman-Filter (KF)
		
		if (mod(t(k),tobs) == 0) % are there observations?
			
			% create observations of the true state 
			YOBS(:,k) = H*XT(:,k)+H*sig_obs*rand(3,1);

			% get the forecast error covariance matrix from the ensemble
			D = zeros(3,N);
			for iens = 1:N
				D(:,iens) = xfens(iens,:) - mean(xfens,1);
			end
			Pf1 = (1/N)*D*D';

			% localize the covariance matrix?
			if E.localize
				Pf = eye(3).*Pf1;
			else
				Pf = Pf1;
			end

			% update the entire ensemble with the observations
			K_bottom = H*Pf*H' + R;
			K_top = H*Pf';
			K = K_top * inv(K_bottom);
			for iens = 1:N
				% for each ensemble member, create a perturbed observation vector
				yens = YOBS(:,k)+H*sig_obs*rand(3,1);
				XENS(iens,:,k) = xfens(iens,:)' + K*(yens - H*xfens(iens,:)');
            end

        else
            % if no observation, then the forecast becomes the analysis
            XENS(:,:,k) = xfens;
        end
        
    else
	    % if not running the filter, then the forecast is the analysis
	    XENS(:,:,k) = xfens; %[v] same val @ this time 'k' as init val line98
	end


	% regardless of whether there's been an observation, the analysis error
    % - covariance matrix comes from the ensemble
	D = zeros(3,N); %[v] for THIS 'k' time, changes every 'k' iter
	for iens = 1:N %[vasu] for each ensemble 'iens' in total 'N'
        D(:,iens) = XENS(iens,:,k) - mean(XENS(:,:,k),1);
	end
	Pa = (1/N)*(D*D');
 
	% save the diagonals of the analysis error covariance matrix -- 
    % - these are the variances
	S(:,k) = diag(Pa);
    
    %[v] w/ dynamical model = lorenz63 (M, cf. 2022_Pirk, eq.2, pg.3):
	% evolve the "truth" forward
	XT(:,k+1) = lorenz63(XT(:,k), sigma, rho, beta, dt);

	% evolve the "analysis ensemble" forward to become the next forecast
	xfens = zeros(N,3);
	for iens = 1:N %[v] for each ensemble, evolve forecast
        xfens(iens,:)  = lorenz63(XENS(iens,:,k), sigma, rho, beta, dt);
	end
end

%---------------PLOTTING----------------------------------

%% Compute a few other output quantities
XA = squeeze(mean(XENS,1)); %[vasu] mean across all the ensembles, 1-dim

%% Produce Plots!
YL = {'x','y','z'};

%% definte some plot settings
LW = 2;		% line width
tcol = [0,0,0];
% acol = [217,95,2]/256.0; %[vasu] commented, repeated line below
acol = [102,166,30]/256.0;
ocol = [241,41,138]/256.0;
ecol = .7*ones(1,3);


% plot the state analysis versis truth
figure(1),clf
h = zeros(1,4);  % legend handle
T = ones(N,1)*t;
for ic = 1:3
  subplot(3,1,ic)
    ens = transpose(squeeze(XENS(:,ic,:))); %[v] from all ens, with ic state
    dum  = plot(transpose(T),ens,'Color',ecol,'LineWidth',1);
    hold on
    h(1) = dum(1);
    h(2) = plot(t,XT(ic,:),'Color',tcol,'LineWidth',LW);
    h(3) = plot(t,XA(ic,:),'Color',acol,'LineWidth',LW);
    h(4) = plot(t,YOBS(ic,:),'o','Color',ocol,'MarkerSize',5,'LineWidth',LW);
    if ic == 1
	    title('Lorenz 1963 Model - State Variables')
    end
    if ic == 3
	    xlabel('time')
	    legend(h, 'ensemble','truth','analysis','obs','Location','SouthOutside','Orientation','Horizontal')
    end
    ylabel(YL(ic))
end

% plot the estimated and true errors
ET = sqrt((XT-XA).^2); %[v] RMSE
EA = sqrt(S);

figure(2),clf
for ic = 1:3
  subplot(3,1,ic)
    h = zeros(1,2);
    h(1) = semilogy(t,ET(ic,:),'Color',tcol,'LineWidth',LW);
    hold on
    h(2) = semilogy(t,EA(ic,:),'Color',acol,'LineWidth',LW);
    ylabel(YL(ic))
    if ic == 1
	    title('Lorenz 1963 Model - RMSE')
    end
    if ic == 3
	    xlabel('time')
	    legend(h, 't-a RMSE','a-err covar RMSE','Location','SouthOutside','Orientation','Horizontal')
    end
end

% average the errors for each variable over the integration time, and print to the screent
ETave = mean(ET,2); %[v] changed from 'nanmean' (not recommend) to 'mean'
EAave = mean(EA,2);
names = {'x','y','z'};
disp('++++++++++ Assimilation Run Average Errors ++++++++++')
for ic = 1:3
	str_out = strcat(names(ic),':	True Error = ',num2str(ET(ic),3),...
        '  Estimated = ',num2str(EA(ic),3));
	disp(str_out)
end
disp('Average State Error')
str_out = strcat('True Error = ',num2str(mean(ETave),3),...
    '	Estimated = ',num2str(mean(EAave),3));
disp(str_out)


% pack everything into the output structure  
A = struct('xt',XT(1,:),...
	'yt',XT(2,:),...
	'zt',XT(3,:),...
	'xa',XA(1,:),...
	'ya',XA(2,:),...
	'za',XA(3,:),...
	'xo',YOBS(1,:),...
	'yo',YOBS(2,:),...
	'zo',YOBS(3,:),...
	'ETave',ETave,...
	'EAave',EAave);

%[vasu]
figure(3); plot3(A.xt,A.yt,A.zt,'Color', tcol); grid;
hold on; plot3(A.xa,A.ya,A.za,'Color', acol);

end