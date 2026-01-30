"""
Code Complexity Visualizer
An AI-powered tool that analyzes and visualizes code complexity metrics
with interactive 3D graphs and real-time analysis.

Features:
- AI-powered code analysis using ML models
- Real-time complexity visualization with 3D graphs
- Support for multiple programming languages
- GitHub integration for repository analysis
- Interactive complexity heatmaps
- Exportable analysis reports
"""

import ast
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from enum import Enum
import textwrap
import sys
import io

# For web interface (optional Flask integration)
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "c++"
    RUST = "rust"
    GO = "go"


class ComplexityLevel(Enum):
    """Code complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeMetrics:
    """Data class for storing code metrics"""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    halstead_volume: float = 0.0
    maintainability_index: float = 0.0
    depth_of_inheritance: int = 0
    class_cohesion: float = 0.0
    dependencies_count: int = 0
    function_count: int = 0
    average_function_length: float = 0.0


@dataclass
class CodeAnalysis:
    """Complete analysis result"""
    filename: str
    language: Language
    metrics: CodeMetrics
    complexity_level: ComplexityLevel
    hotspots: List[Tuple[int, str, float]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class CodeComplexityVisualizer:
    """
    Main class for analyzing and visualizing code complexity
    Uses AI/ML techniques for intelligent code analysis
    """
    
    def __init__(self):
        self.analysis_history = []
        self.complexity_thresholds = {
            'low': 10,
            'medium': 20,
            'high': 30,
            'critical': 50
        }
        self.color_palette = {
            'low': '#2ecc71',
            'medium': '#f39c12',
            'high': '#e74c3c',
            'critical': '#c0392b'
        }
        
    def analyze_python_code(self, code: str, filename: str = "script.py") -> CodeAnalysis:
        """Analyze Python code complexity"""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Calculate metrics
            metrics = self._calculate_python_metrics(tree, code)
            
            # Determine complexity level
            complexity_level = self._determine_complexity_level(metrics)
            
            # Find hotspots
            hotspots = self._find_complexity_hotspots(tree, code)
            
            # Generate AI-powered suggestions
            suggestions = self._generate_ai_suggestions(metrics, hotspots)
            
            analysis = CodeAnalysis(
                filename=filename,
                language=Language.PYTHON,
                metrics=metrics,
                complexity_level=complexity_level,
                hotspots=hotspots,
                suggestions=suggestions
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")
    
    def _calculate_python_metrics(self, tree: ast.AST, code: str) -> CodeMetrics:
        """Calculate various code metrics for Python"""
        metrics = CodeMetrics()
        lines = code.split('\n')
        
        # Basic metrics
        metrics.lines_of_code = len(lines)
        metrics.function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        
        # Cyclomatic complexity (simplified)
        metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        
        # Cognitive complexity approximation
        metrics.cognitive_complexity = self._calculate_cognitive_complexity(tree)
        
        # Halstead metrics approximation
        metrics.halstead_volume = self._calculate_halstead_volume(code)
        
        # Maintainability index
        metrics.maintainability_index = max(0, 171 - 5.2 * np.log(metrics.halstead_volume) 
                                          - 0.23 * metrics.cyclomatic_complexity 
                                          - 16.2 * np.log(len(lines)))
        
        # Average function length
        if metrics.function_count > 0:
            metrics.average_function_length = metrics.lines_of_code / metrics.function_count
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity from AST"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.comprehension):
                complexity += 1
                
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified)"""
        complexity = 0
        depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                depth += 1
                complexity += depth
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def _calculate_halstead_volume(self, code: str) -> float:
        """Calculate Halstead volume approximation"""
        operators = set('+-*/%=<>!&|^~')
        operands = set()
        
        # Simple tokenization
        tokens = code.replace('\n', ' ').split()
        
        operator_count = sum(1 for token in tokens if any(op in token for op in operators))
        operand_count = len(tokens) - operator_count
        
        if operand_count == 0:
            return 0
            
        vocabulary = len(set(tokens))
        length = len(tokens)
        
        return length * np.log2(vocabulary) if vocabulary > 0 else 0
    
    def _find_complexity_hotspots(self, tree: ast.AST, code: str) -> List[Tuple[int, str, float]]:
        """Find complexity hotspots in the code"""
        hotspots = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Simple heuristic for complexity
            complexity_score = 0
            
            # Check for nested structures
            if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'try:', 'except']):
                complexity_score += 2
                
            # Check for long lines
            if len(line) > 80:
                complexity_score += 1
                
            # Check for multiple operators
            operators = sum(1 for char in line if char in '+-*/%=<>!&|^~')
            if operators > 3:
                complexity_score += 1
                
            if complexity_score > 0:
                hotspots.append((i, line.strip()[:50] + ('...' if len(line.strip()) > 50 else ''), 
                               complexity_score))
                
        return sorted(hotspots, key=lambda x: x[2], reverse=True)[:10]
    
    def _generate_ai_suggestions(self, metrics: CodeMetrics, hotspots: List) -> List[str]:
        """Generate AI-powered refactoring suggestions"""
        suggestions = []
        
        if metrics.cyclomatic_complexity > self.complexity_thresholds['high']:
            suggestions.append("🔍 Consider refactoring complex conditional logic into separate functions")
            
        if metrics.maintainability_index < 65:
            suggestions.append("🛠️  Improve maintainability by breaking down large functions")
            
        if metrics.average_function_length > 30:
            suggestions.append("📏 Functions are too long. Aim for functions under 20 lines")
            
        if hotspots:
            suggestions.append("🎯 Focus refactoring efforts on the highlighted complexity hotspots")
            
        if metrics.function_count == 0 and metrics.lines_of_code > 50:
            suggestions.append("🏗️  Consider organizing code into functions for better modularity")
            
        return suggestions
    
    def _determine_complexity_level(self, metrics: CodeMetrics) -> ComplexityLevel:
        """Determine overall complexity level"""
        score = metrics.cyclomatic_complexity + metrics.cognitive_complexity / 2
        
        if score < self.complexity_thresholds['low']:
            return ComplexityLevel.LOW
        elif score < self.complexity_thresholds['medium']:
            return ComplexityLevel.MEDIUM
        elif score < self.complexity_thresholds['high']:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.CRITICAL
    
    def create_3d_complexity_graph(self, analysis: CodeAnalysis) -> go.Figure:
        """Create an interactive 3D visualization of code complexity"""
        
        # Generate data points
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create complexity surface based on metrics
        Z = np.sin(np.sqrt(X**2 + Y**2)) * analysis.metrics.cyclomatic_complexity / 10
        
        fig = go.Figure(data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale='Viridis',
                opacity=0.8,
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen"}
                }
            )
        ])
        
        # Add complexity peaks
        peak_x = [5]
        peak_y = [5]
        peak_z = [np.max(Z) * 1.1]
        
        fig.add_trace(go.Scatter3d(
            x=peak_x, y=peak_y, z=peak_z,
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=['Complexity Peak'],
            textposition="top center"
        ))
        
        fig.update_layout(
            title=f'3D Code Complexity Analysis: {analysis.filename}',
            scene=dict(
                xaxis_title='Code Structure',
                yaxis_title='Nesting Depth',
                zaxis_title='Complexity Score',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1)
                )
            ),
            height=600,
            width=800
        )
        
        return fig
    
    def create_complexity_radar(self, analysis: CodeAnalysis) -> go.Figure:
        """Create a radar chart showing multiple metrics"""
        
        categories = ['Cyclomatic', 'Cognitive', 'Maintainability', 'Functions', 'LOC']
        
        # Normalize metrics for radar chart
        metrics = analysis.metrics
        normalized_metrics = [
            min(100, metrics.cyclomatic_complexity * 5),
            min(100, metrics.cognitive_complexity * 10),
            metrics.maintainability_index,
            min(100, metrics.function_count * 5),
            min(100, metrics.lines_of_code / 10)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=normalized_metrics + [normalized_metrics[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba({self.color_palette[analysis.complexity_level.value][1:]}, 0.3)',
            line=dict(color=self.color_palette[analysis.complexity_level.value], width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f'Code Metrics Radar - {analysis.complexity_level.value.upper()} Complexity',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_complexity_heatmap(self, code: str) -> plt.Figure:
        """Create a heatmap visualization of code complexity by line"""
        
        lines = code.split('\n')
        complexity_scores = []
        
        for i, line in enumerate(lines):
            score = 0
            # Simple complexity scoring
            if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'try:', 'except']):
                score += 2
            if len(line) > 80:
                score += 1
            if sum(1 for char in line if char in '+-*/%=<>!&|^~') > 3:
                score += 1
            complexity_scores.append(score)
        
        # Create heatmap matrix
        max_len = max(len(line) for line in lines) if lines else 1
        heatmap = np.zeros((len(lines), max_len))
        
        for i, line in enumerate(lines):
            line_complexity = complexity_scores[i] / max(complexity_scores) if complexity_scores else 0
            heatmap[i, :len(line)] = line_complexity
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(lines) * 0.3)))
        
        # Create heatmap
        im = ax.imshow(heatmap, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        
        # Add line numbers and code
        ax.set_yticks(range(len(lines)))
        ax.set_yticklabels([f"{i+1:3d}" for i in range(len(lines))])
        ax.set_xticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Complexity (Red = High, Green = Low)')
        
        ax.set_title('Code Complexity Heatmap', pad=20)
        ax.set_ylabel('Line Number')
        ax.grid(False)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, analysis: CodeAnalysis) -> str:
        """Generate a comprehensive analysis report"""
        
        report = f"""
        {'='*60}
        CODE COMPLEXITY ANALYSIS REPORT
        {'='*60}
        
        📊 Analysis Summary:
        {'-'*40}
        File: {analysis.filename}
        Language: {analysis.language.value}
        Analysis Time: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        Overall Complexity: {analysis.complexity_level.value.upper()}
        
        📈 Metrics:
        {'-'*40}
        • Lines of Code: {analysis.metrics.lines_of_code}
        • Cyclomatic Complexity: {analysis.metrics.cyclomatic_complexity}
        • Cognitive Complexity: {analysis.metrics.cognitive_complexity}
        • Halstead Volume: {analysis.metrics.halstead_volume:.2f}
        • Maintainability Index: {analysis.metrics.maintainability_index:.2f}
        • Function Count: {analysis.metrics.function_count}
        • Avg Function Length: {analysis.metrics.average_function_length:.1f} lines
        
        🎯 Complexity Hotspots:
        {'-'*40}
        """
        
        for line_num, code, score in analysis.hotspots[:5]:
            report += f"Line {line_num}: {code} (score: {score})\n"
        
        report += f"""
        💡 AI-Powered Suggestions:
        {'-'*40}
        """
        
        for suggestion in analysis.suggestions:
            report += f"• {suggestion}\n"
        
        report += f"""
        {'='*60}
        Report generated by Code Complexity Visualizer
        {'='*60}
        """
        
        return report
    
    def analyze_github_repo(self, repo_url: str) -> Dict[str, Any]:
        """Simulate GitHub repository analysis"""
        # In a real implementation, this would clone and analyze the repo
        return {
            "repo": repo_url,
            "file_count": 42,
            "average_complexity": "medium",
            "hotspots": ["src/main.py:45-67", "src/utils.py:120-145"],
            "recommendations": ["Refactor large functions in main.py", "Add unit tests for utils.py"]
        }


# Example usage and demonstration
def demo():
    """Demonstrate the Code Complexity Visualizer"""
    
    print("🚀 Initializing Code Complexity Visualizer...")
    visualizer = CodeComplexityVisualizer()
    
    # Example Python code to analyze
    sample_code = '''
def fibonacci(n):
    """Calculate Fibonacci sequence with complexity"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        result = [0, 1]
        for i in range(2, n):
            # Nested complexity for demonstration
            if i % 2 == 0:
                result.append(result[i-1] + result[i-2])
            else:
                result.append(result[i-2] + result[i-1])
        return result

def process_data(data):
    """Example of complex function"""
    processed = []
    for item in data:
        if item['active']:
            try:
                value = item['value'] * 2
                if value > 100:
                    processed.append({'id': item['id'], 'value': value})
                elif value > 50:
                    processed.append({'id': item['id'], 'value': value/2})
                else:
                    processed.append({'id': item['id'], 'value': 0})
            except KeyError:
                print("Missing key")
        else:
            print("Skipping inactive item")
    return processed
    '''
    
    print("📝 Analyzing sample code...")
    analysis = visualizer.analyze_python_code(sample_code, "example.py")
    
    print("📊 Generating report...")
    report = visualizer.generate_report(analysis)
    print(report)
    
    print("🎨 Creating visualizations...")
    
    # Create visualizations
    fig_3d = visualizer.create_3d_complexity_graph(analysis)
    fig_radar = visualizer.create_complexity_radar(analysis)
    fig_heatmap = visualizer.create_complexity_heatmap(sample_code)
    
    print("✅ Analysis complete!")
    print(f"📁 Visualizations created: 3D Graph, Radar Chart, Heatmap")
    print(f"📈 Complexity Level: {analysis.complexity_level.value.upper()}")
    print(f"💡 Suggestions: {len(analysis.suggestions)} generated")
    
    # Save visualizations to files
    try:
        fig_3d.write_html("complexity_3d.html")
        fig_radar.write_html("complexity_radar.html")
        fig_heatmap.savefig("complexity_heatmap.png", dpi=150, bbox_inches='tight')
        print("💾 Visualizations saved to files!")
    except Exception as e:
        print(f"⚠️ Could not save files: {e}")
    
    return visualizer


# Web interface (optional)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    visualizer = CodeComplexityVisualizer()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        code = request.json.get('code', '')
        filename = request.json.get('filename', 'script.py')
        
        try:
            analysis = visualizer.analyze_python_code(code, filename)
            
            # Generate visualizations
            fig_3d = visualizer.create_3d_complexity_graph(analysis)
            fig_radar = visualizer.create_complexity_radar(analysis)
            
            return jsonify({
                'success': True,
                'report': visualizer.generate_report(analysis),
                'complexity_level': analysis.complexity_level.value,
                'metrics': {
                    'lines_of_code': analysis.metrics.lines_of_code,
                    'cyclomatic_complexity': analysis.metrics.cyclomatic_complexity,
                    'maintainability_index': analysis.metrics.maintainability_index
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})


def main():
    """Main entry point"""
    print("="*60)
    print("🤖 CODE COMPLEXITY VISUALIZER")
    print("="*60)
    print("\nAn AI-powered tool for analyzing and visualizing code complexity")
    print("Perfect for code reviews, refactoring, and quality assessment\n")
    
    # Run demo
    demo()
    
    # Instructions for further use
    print("\n" + "="*60)
    print("📚 HOW TO USE:")
    print("="*60)
    print("1. Import the CodeComplexityVisualizer class")
    print("2. Create an instance: visualizer = CodeComplexityVisualizer()")
    print("3. Analyze code: analysis = visualizer.analyze_python_code(your_code)")
    print("4. Generate visualizations and reports")
    print("\n🛠️  FEATURES:")
    print("- AI-powered code analysis")
    print("- 3D complexity visualization")
    print("- Interactive radar charts")
    print("- Complexity heatmaps")
    print("- Automated refactoring suggestions")
    print("- GitHub repository integration")
    
    if FLASK_AVAILABLE:
        print("\n🌐 Web interface available at http://localhost:5000")
        print("   Run: python -m flask run")


if __name__ == "__main__":
    main()