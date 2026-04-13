import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  subValue?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'default' | 'success' | 'danger' | 'warning';
}

const StatsCard: React.FC<StatsCardProps> = ({ 
  title, 
  value, 
  subValue, 
  icon: Icon,
  color = 'default' 
}) => {
  
  const colorClasses = {
    default: 'text-neutral-100',
    success: 'text-emerald-400',
    danger: 'text-rose-400',
    warning: 'text-amber-400'
  };

  return (
    <div className="bg-surface rounded-xl p-5 border border-neutral-800 shadow-sm flex items-start justify-between hover:border-neutral-600 transition-colors h-full">
      <div>
        <p className="text-neutral-400 text-sm font-medium mb-1">{title}</p>
        <h3 className={`text-2xl font-bold ${colorClasses[color]}`}>{value}</h3>
        {subValue && <p className="text-neutral-500 text-xs mt-1">{subValue}</p>}
      </div>
      {Icon && (
        <div className={`p-2 rounded-lg bg-neutral-800 ${colorClasses[color].replace('text-', 'bg-').replace('400', '500')}/10`}>
          <Icon className={`w-5 h-5 ${colorClasses[color]}`} />
        </div>
      )}
    </div>
  );
};

export default StatsCard;