import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'capture_screen.dart';

class HeightScreen extends StatefulWidget {
  const HeightScreen({super.key});

  @override
  State<HeightScreen> createState() => _HeightScreenState();
}

class _HeightScreenState extends State<HeightScreen> {
  final _controller = TextEditingController();
  bool _isValid = false;

  static const double minHeight = 100;
  static const double maxHeight = 220;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  String? _validate(String? value) {
    if (value == null || value.trim().isEmpty) {
      return 'Please enter your height';
    }
    final h = double.tryParse(value.trim());
    if (h == null) {
      return 'Enter a valid number';
    }
    if (h < minHeight || h > maxHeight) {
      return 'Height should be between ${minHeight.toInt()} and ${maxHeight.toInt()} cm';
    }
    return null;
  }

  void _onContinue() {
    final height = double.parse(_controller.text.trim());
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CaptureScreen(heightCm: height),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 60),
              const Text(
                "What's your height?",
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              const Text(
                'We use it to work out your measurements from the photo.',
              ),
              const SizedBox(height: 32),
              TextFormField(
                controller: _controller,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'^\d{0,3}\.?\d?')),
                ],
                autovalidateMode: AutovalidateMode.onUserInteraction,
                validator: _validate,
                onChanged: (value) {
                  setState(() {
                    _isValid = _validate(value) == null;
                  });
                },
                decoration: const InputDecoration(
                  labelText: 'Height',
                  suffixText: 'cm',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: _isValid ? _onContinue : null,
                  child: const Text('Continue'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}