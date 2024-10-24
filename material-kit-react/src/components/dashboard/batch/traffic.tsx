'use client';
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { Chart } from '@/components/core/chart';
import { useTheme } from '@mui/material/styles';
import type { ApexOptions } from 'apexcharts';

export interface TrafficProps {
  chartSeries: number[];
  labels: string[];
}

export function Traffic({ chartSeries, labels }: TrafficProps): React.JSX.Element {
  const chartOptions = useChartOptions(labels);

  return (
    <Card>
      <CardHeader title="Traffic Type by Packets" />
      <CardContent>
        <Stack spacing={2}>
          <Chart height={200} options={chartOptions} series={chartSeries} type="donut" width="100%" />
          <Stack direction="row" spacing={2} sx={{ alignItems: 'center', justifyContent: 'center' }}>
            {chartSeries.map((item, index) => {
              const label = labels[index];
              return (
                <Stack key={label} spacing={1} sx={{ alignItems: 'center' }}>
                  <Typography variant="h6">{label}</Typography>
                  <Typography color="text.secondary" variant="subtitle2">
                    {item.toFixed(2)}%
                  </Typography>
                </Stack>
              );
            })}
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}

function useChartOptions(labels: string[]): ApexOptions {
  const theme = useTheme();

  return {
    chart: { background: 'transparent' },
    colors: [theme.palette.primary.main, theme.palette.success.main, theme.palette.warning.main],
    dataLabels: { enabled: false },
    labels,
    legend: { show: false },
    plotOptions: { pie: { expandOnClick: false } },
    states: { active: { filter: { type: 'none' } }, hover: { filter: { type: 'none' } } },
    stroke: { width: 0 },
    theme: { mode: theme.palette.mode },
    tooltip: { fillSeriesColor: false },
  };
}
