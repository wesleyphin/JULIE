import React from 'react';
import { PropFirmAggregateStats } from '../types';
import { Users, AlertCircle, CheckCircle2, Trophy, Banknote, ArrowRight, Skull } from 'lucide-react';

interface Props {
  stats: PropFirmAggregateStats;
}

const StageCard = ({ title, count, total, color, icon: Icon, subtext }: any) => (
  <div className={`bg-surface border border-${color}-900/50 rounded-xl p-4 flex-1 min-w-[200px] relative overflow-hidden`}>
    <div className={`absolute top-0 right-0 p-3 opacity-10 text-${color}-500`}>
        <Icon className="w-16 h-16" />
    </div>
    <div className="relative z-10">
        <h4 className="text-neutral-400 text-sm font-medium uppercase tracking-wider mb-2">{title}</h4>
        <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white">{count.toLocaleString()}</span>
            {total && <span className="text-xs text-neutral-500">/ {total.toLocaleString()} attempts</span>}
        </div>
        {subtext && <p className={`text-xs mt-2 text-${color}-400 font-medium`}>{subtext}</p>}
    </div>
  </div>
);

const MetricBadge = ({ label, value, icon: Icon }: any) => (
  <div className="bg-neutral-900/50 rounded-lg p-3 border border-neutral-700/50 flex flex-col items-center text-center">
    <Icon className="w-5 h-5 text-primary mb-2" />
    <span className="text-xl font-bold text-neutral-200">{value}</span>
    <span className="text-xs text-neutral-500 mt-1">{label}</span>
  </div>
);

const PropFirmDashboard: React.FC<Props> = ({ stats }) => {
  return (
    <div className="space-y-6 animate-in fade-in duration-500">
        
        {/* Funnel Visualization */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative">
            {/* Arrows for Desktop */}
            <div className="hidden md:block absolute top-1/2 left-1/3 -translate-x-1/2 -translate-y-1/2 text-neutral-700 z-0">
                <ArrowRight className="w-8 h-8" />
            </div>
            <div className="hidden md:block absolute top-1/2 left-2/3 -translate-x-1/2 -translate-y-1/2 text-neutral-700 z-0">
                <ArrowRight className="w-8 h-8" />
            </div>

            {/* Stage 1: Eval */}
            <StageCard 
                title="Evaluation Phase" 
                count={stats.evalPassed}
                total={stats.evalAttempts}
                color="blue"
                icon={Users}
                subtext={`${((stats.evalPassed / (stats.evalAttempts || 1)) * 100).toFixed(1)}% Pass Rate`}
            />

            {/* Stage 2: Express */}
            <StageCard 
                title="Express Phase" 
                count={stats.expressAttempts}
                total={null} 
                color="amber"
                icon={Banknote}
                subtext={`${stats.expressPayoutsTotal.toLocaleString()} Payouts Generated`}
            />

            {/* Stage 3: Live */}
            <StageCard 
                title="Live Funded" 
                count={stats.liveReached}
                total={stats.expressAttempts}
                color="emerald"
                icon={Trophy}
                subtext={`${stats.expressPassRate.toFixed(1)}% Conversion Rate`}
            />
        </div>

        {/* Detailed Stats Grid */}
        <div className="bg-surface rounded-xl border border-neutral-700 p-6">
            <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-secondary" />
                Failure & Efficiency Analysis
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                
                {/* Blowups */}
                <div className="space-y-4">
                    <h4 className="text-sm text-neutral-400 font-medium border-b border-neutral-700 pb-2">Account Blowups</h4>
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-neutral-300">Evaluation Phase</span>
                        <span className="text-rose-400 font-mono font-bold">{stats.evalBlown.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-neutral-300">Express Phase</span>
                        <span className="text-rose-400 font-mono font-bold">{stats.expressBlown.toLocaleString()}</span>
                    </div>
                    
                    {stats.blownAfterFirstPayout > 0 && (
                        <div className="flex justify-between items-center pl-2 border-l-2 border-neutral-800">
                             <span className="text-xs text-neutral-500">After 1st Payout</span>
                             <span className="text-xs text-rose-500 font-mono">{stats.blownAfterFirstPayout.toLocaleString()}</span>
                        </div>
                    )}

                    <div className="mt-2 pt-2 border-t border-neutral-800">
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-neutral-500">Total Failures</span>
                            <span className="text-rose-500 font-mono font-bold">{(stats.evalBlown + stats.expressBlown).toLocaleString()}</span>
                        </div>
                    </div>
                </div>

                {/* Efficiency */}
                <div className="lg:col-span-3 grid grid-cols-2 lg:grid-cols-3 gap-4">
                    <MetricBadge 
                        label="Avg Trials to Express" 
                        value={stats.avgTrialsToExpress.toFixed(1)}
                        icon={CheckCircle2}
                    />
                    <MetricBadge 
                        label="Avg Trials to 1st Payout" 
                        value={stats.avgTrialsToFirstPayout.toFixed(1)}
                        icon={Banknote}
                    />
                    <MetricBadge 
                        label="Avg Trials to Live" 
                        value={stats.avgTrialsToLive.toFixed(1)}
                        icon={Trophy}
                    />
                </div>
            </div>
        </div>

        <div className="bg-neutral-900/30 rounded-lg p-4 text-xs text-neutral-500 border border-neutral-800">
            <p><strong>Simulation Logic:</strong> {stats.totalTraders} virtual traders simulated. Accounts restart from Evaluation (Step 1) immediately upon failure. Max career duration capped to prevent infinite loops. Metrics "Avg Trials" count the number of fresh starts required to achieve the milestone.</p>
        </div>

    </div>
  );
};

export default PropFirmDashboard;