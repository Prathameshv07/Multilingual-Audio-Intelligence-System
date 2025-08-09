"""
Advanced Visualization Components for Multilingual Audio Intelligence System

This module provides sophisticated visualization components for creating
interactive audio analysis interfaces. Features include waveform visualization,
speaker timelines, and processing feedback displays.

Key Features:
- Interactive waveform with speaker segment overlays
- Speaker activity timeline visualization
- Processing progress indicators
- Exportable visualizations

Dependencies: plotly, matplotlib, numpy
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
import base64
import io
from datetime import datetime
import json

# Safe imports with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Some visualizations will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Fallback visualizations will be used.")

logger = logging.getLogger(__name__)


class WaveformVisualizer:
    """Advanced waveform visualization with speaker overlays."""
    
    def __init__(self, width: int = 1000, height: int = 300):
        self.width = width
        self.height = height
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def create_interactive_waveform(self, 
                                  audio_data: np.ndarray,
                                  sample_rate: int,
                                  speaker_segments: List[Dict],
                                  transcription_segments: List[Dict] = None):
        """
        Create interactive waveform visualization with speaker overlays.
        
        Args:
            audio_data: Audio waveform data
            sample_rate: Audio sample rate
            speaker_segments: List of speaker segment dicts
            transcription_segments: Optional transcription data
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_visualization(audio_data, sample_rate, speaker_segments)
        
        try:
            # Create time axis
            time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
            
            # Downsample for visualization if needed
            if len(audio_data) > 50000:
                step = len(audio_data) // 50000
                audio_data = audio_data[::step]
                time_axis = time_axis[::step]
            
            # Create the main plot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Audio Waveform with Speaker Segments", "Speaker Timeline"),
                vertical_spacing=0.1
            )
            
            # Add waveform
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=audio_data,
                    mode='lines',
                    name='Waveform',
                    line=dict(color='#2C3E50', width=1),
                    hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add speaker segment overlays
            speaker_colors = {}
            for i, segment in enumerate(speaker_segments):
                speaker_id = segment.get('speaker_id', f'Speaker_{i}')
                
                if speaker_id not in speaker_colors:
                    speaker_colors[speaker_id] = self.colors[len(speaker_colors) % len(self.colors)]
                
                # Add shaded region for speaker segment
                fig.add_vrect(
                    x0=segment['start_time'],
                    x1=segment['end_time'],
                    fillcolor=speaker_colors[speaker_id],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
                
                # Add speaker label
                mid_time = (segment['start_time'] + segment['end_time']) / 2
                if len(audio_data) > 0:
                    fig.add_annotation(
                        x=mid_time,
                        y=max(audio_data) * 0.8,
                        text=speaker_id.replace('SPEAKER_', 'S'),
                        showarrow=False,
                        font=dict(color=speaker_colors[speaker_id], size=10, family="Arial Black"),
                        row=1, col=1
                    )
            
            # Create speaker timeline in bottom subplot
            for i, (speaker_id, color) in enumerate(speaker_colors.items()):
                speaker_segments_filtered = [s for s in speaker_segments if s['speaker_id'] == speaker_id]
                
                for segment in speaker_segments_filtered:
                    fig.add_trace(
                        go.Scatter(
                            x=[segment['start_time'], segment['end_time']],
                            y=[i, i],
                            mode='lines',
                            name=speaker_id,
                            line=dict(color=color, width=8),
                            showlegend=(segment == speaker_segments_filtered[0]),
                            hovertemplate=f'{speaker_id}<br>%{{x:.2f}}s<extra></extra>'
                        ),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="🎵 Multilingual Audio Intelligence Visualization",
                    font=dict(size=20, family="Arial Black"),
                    x=0.5
                ),
                height=600,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='#F8F9FA'
            )
            
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            if speaker_colors:
                fig.update_yaxes(title_text="Speaker", row=2, col=1, 
                               ticktext=list(speaker_colors.keys()), 
                               tickvals=list(range(len(speaker_colors))))
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating waveform visualization: {e}")
            return self._create_fallback_visualization(audio_data, sample_rate, speaker_segments)
    
    def _create_fallback_visualization(self, audio_data, sample_rate, speaker_segments):
        """Create a simple fallback visualization when Plotly is not available."""
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(
                text="Waveform visualization temporarily unavailable",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Audio Waveform Visualization",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude"
            )
            return fig
        else:
            # Return a simple HTML representation
            return None
    
    def create_language_distribution_chart(self, segments: List[Dict]):
        """Create language distribution visualization."""
        if not PLOTLY_AVAILABLE:
            return None
            
        try:
            # Count languages
            language_counts = {}
            language_durations = {}
            
            for segment in segments:
                lang = segment.get('original_language', 'unknown')
                duration = segment.get('end_time', 0) - segment.get('start_time', 0)
                
                language_counts[lang] = language_counts.get(lang, 0) + 1
                language_durations[lang] = language_durations.get(lang, 0) + duration
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Language Distribution by Segments', 'Language Distribution by Duration'),
                specs=[[{'type': 'domain'}, {'type': 'domain'}]]
            )
            
            # Pie chart for segment counts
            fig.add_trace(
                go.Pie(
                    labels=list(language_counts.keys()),
                    values=list(language_counts.values()),
                    name="Segments",
                    hovertemplate='%{label}<br>%{value} segments<br>%{percent}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Pie chart for durations
            fig.add_trace(
                go.Pie(
                    labels=list(language_durations.keys()),
                    values=list(language_durations.values()),
                    name="Duration",
                    hovertemplate='%{label}<br>%{value:.1f}s<br>%{percent}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="🌍 Language Analysis",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating language distribution chart: {e}")
            return None


class SubtitleRenderer:
    """Advanced subtitle rendering with synchronization."""
    
    def __init__(self):
        self.subtitle_style = """
        <style>
        .subtitle-container {
            max-height: 400px;
            overflow-y: auto;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
        .subtitle-segment {
            background: rgba(255,255,255,0.95);
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4ECDC4;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .subtitle-segment:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .subtitle-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .speaker-label {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .timestamp {
            color: #666;
            font-size: 12px;
            font-family: 'Courier New', monospace;
        }
        .language-tag {
            background: #45B7D1;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            margin-left: 5px;
        }
        .original-text {
            margin: 8px 0;
            font-size: 16px;
            color: #2C3E50;
            line-height: 1.4;
        }
        .translated-text {
            margin: 8px 0;
            font-size: 14px;
            color: #7F8C8D;
            font-style: italic;
            line-height: 1.4;
            border-top: 1px solid #ECF0F1;
            padding-top: 8px;
        }
        .confidence-bar {
            width: 100%;
            height: 4px;
            background: #ECF0F1;
            border-radius: 2px;
            overflow: hidden;
            margin-top: 5px;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
            transition: width 0.3s ease;
        }
        </style>
        """
    
    def render_subtitles(self, segments: List[Dict], show_translations: bool = True) -> str:
        """
        Render beautiful HTML subtitles with speaker attribution.
        
        Args:
            segments: List of processed segments
            show_translations: Whether to show translations
            
        Returns:
            str: HTML formatted subtitles
        """
        try:
            html_parts = [self.subtitle_style]
            html_parts.append('<div class="subtitle-container">')
            
            for i, segment in enumerate(segments):
                speaker_id = segment.get('speaker_id', f'Speaker_{i}')
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                original_text = segment.get('original_text', '')
                translated_text = segment.get('translated_text', '')
                original_language = segment.get('original_language', 'unknown')
                confidence = segment.get('confidence_transcription', 0.0)
                
                # Format timestamps
                start_str = self._format_timestamp(start_time)
                end_str = self._format_timestamp(end_time)
                
                html_parts.append('<div class="subtitle-segment">')
                
                # Header with speaker and timestamp
                html_parts.append('<div class="subtitle-header">')
                html_parts.append(f'<span class="speaker-label">{speaker_id.replace("SPEAKER_", "Speaker ")}</span>')
                html_parts.append(f'<span class="timestamp">{start_str} - {end_str}</span>')
                html_parts.append('</div>')
                
                # Original text with language tag
                if original_text:
                    html_parts.append('<div class="original-text">')
                    html_parts.append(f'🗣️ {original_text}')
                    html_parts.append(f'<span class="language-tag">{original_language.upper()}</span>')
                    html_parts.append('</div>')
                
                # Translated text
                if show_translations and translated_text and translated_text != original_text:
                    html_parts.append('<div class="translated-text">')
                    html_parts.append(f'🔄 {translated_text}')
                    html_parts.append('</div>')
                
                # Confidence indicator
                confidence_percent = confidence * 100
                html_parts.append('<div class="confidence-bar">')
                html_parts.append(f'<div class="confidence-fill" style="width: {confidence_percent}%"></div>')
                html_parts.append('</div>')
                
                html_parts.append('</div>')
            
            html_parts.append('</div>')
            return ''.join(html_parts)
            
        except Exception as e:
            logger.error(f"Error rendering subtitles: {e}")
            return f'<div style="color: red; padding: 20px;">Error rendering subtitles: {str(e)}</div>'
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format."""
        try:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}:{secs:05.2f}"
        except:
            return "00:00.00"


class PerformanceMonitor:
    """Real-time performance monitoring component."""
    
    def create_performance_dashboard(self, processing_stats: Dict) -> str:
        """Create performance monitoring dashboard."""
        try:
            component_times = processing_stats.get('component_times', {})
            total_time = processing_stats.get('total_time', 0)
            
            if PLOTLY_AVAILABLE and component_times:
                # Create performance chart
                components = list(component_times.keys())
                times = list(component_times.values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=components,
                        y=times,
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(components)],
                        text=[f'{t:.2f}s' for t in times],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title='⚡ Processing Performance Breakdown',
                    xaxis_title='Pipeline Components',
                    yaxis_title='Processing Time (seconds)',
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='#F8F9FA'
                )
                
                # Convert to HTML
                plot_html = fig.to_html(include_plotlyjs='cdn', div_id='performance-chart')
            else:
                plot_html = '<div style="text-align: center; padding: 40px;">Performance chart temporarily unavailable</div>'
            
            # Add summary stats
            stats_html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 15px; margin: 10px 0;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                <h3 style="margin: 0 0 15px 0;">📊 Processing Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 24px; font-weight: bold;">{total_time:.2f}s</div>
                        <div style="opacity: 0.8;">Total Processing Time</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 24px; font-weight: bold;">{processing_stats.get('num_speakers', 0)}</div>
                        <div style="opacity: 0.8;">Speakers Detected</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 24px; font-weight: bold;">{processing_stats.get('num_segments', 0)}</div>
                        <div style="opacity: 0.8;">Speech Segments</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 24px; font-weight: bold;">{len(processing_stats.get('languages_detected', []))}</div>
                        <div style="opacity: 0.8;">Languages Found</div>
                    </div>
                </div>
            </div>
            """
            
            return stats_html + plot_html
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return f'<div style="color: red;">Performance Dashboard Error: {str(e)}</div>'


class FileDownloader:
    """Enhanced file download component with preview."""
    
    def create_download_section(self, outputs: Dict[str, str], filename_base: str) -> str:
        """Create download section with file previews."""
        download_html = """
        <div style="margin-top: 20px;">
            <h3 style="margin-bottom: 10px;">📥 Download Results</h3>
            <div style="display: flex; flex-direction: column; gap: 10px;">
        """
        
        # Create download buttons for each format
        for format_name, content in outputs.items():
            if format_name in ['json', 'srt_original', 'srt_translated', 'text', 'csv', 'summary']:
                download_html += f"""
                <div style="background: #f0f0f0; padding: 15px; border-radius: 10px; border: 1px solid #ccc;">
                    <h4 style="margin: 0 0 5px 0;">{format_name.upper()} Preview</h4>
                    <pre style="font-size: 14px; white-space: pre-wrap; word-wrap: break-word; background: #fff; padding: 10px; border-radius: 5px; border: 1px solid #eee; overflow-x: auto;">
                        {content[:500]}...
                    </pre>
                    <a href="data:text/{self._get_file_extension(format_name)};base64,{base64.b64encode(content.encode()).decode()}" 
                       download="{filename_base}.{self._get_file_extension(format_name)}" 
                       style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block; margin-top: 10px;">
                        Download {format_name.upper()}
                    </a>
                </div>
                """
        
        download_html += """
            </div>
        </div>
        """
        return download_html
    
    def _get_file_extension(self, format_name: str) -> str:
        """Get appropriate file extension for format."""
        extensions = {
            'json': 'json',
            'srt_original': 'srt',
            'srt_translated': 'en.srt',
            'text': 'txt',
            'csv': 'csv',
            'summary': 'summary.txt'
        }
        return extensions.get(format_name, 'txt')


def create_custom_css() -> str:
    """Create custom CSS for the entire application."""
    return """
    /* Global Styles */
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 30px;
        border-radius: 0 0 20px 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2em;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed #4ECDC4;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(78, 205, 196, 0.1);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #45B7D1;
        background: rgba(69, 183, 209, 0.15);
        transform: translateY(-2px);
    }
    
    /* Button Styles */
    .primary-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .primary-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Card Styles */
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Progress Animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing {
        animation: pulse 1.5s infinite;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2em;
        }
        .main-subtitle {
            font-size: 1em;
        }
    }
    """


def create_loading_animation() -> str:
    """Create loading animation HTML."""
    return """
    <div style="text-align: center; padding: 40px;">
        <div style="display: inline-block; width: 50px; height: 50px; border: 3px solid #f3f3f3; 
                    border-top: 3px solid #4ECDC4; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        <div style="margin-top: 20px; font-size: 18px; color: #666;">
            🎵 Processing your audio with AI magic...
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: #999;">
            This may take a few moments depending on audio length
        </div>
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """


# Export main classes for use in app.py
__all__ = [
    'WaveformVisualizer',
    'SubtitleRenderer', 
    'PerformanceMonitor',
    'FileDownloader',
    'create_custom_css',
    'create_loading_animation'
] 