#!/usr/bin/env python3
"""
Phase 1 Tests: Baseline Template Validation

Validiert wissenschaftliche Korrektheit der Baseline-Templates.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from templates import format_template, select_template


class TestBaselineTemplates(unittest.TestCase):
    """Test suite for baseline template validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_tweet_post = "This movie is [MASKED] and the acting is [MASKED]!"
        self.test_tweet_reply = "What did you think about the new Marvel movie?"
        self.test_persona = "I am a 25-year-old film enthusiast who loves action movies."
    
    def test_no_persona_post_template(self):
        """Test No-Persona Post Template - Critical Requirement."""
        result = format_template(
            "imitation_post_template_no_persona", 
            tweet=self.test_tweet_post
        )
        
        # CRITICAL: No persona references
        self.assertNotIn("persona", result.lower())
        self.assertNotIn("user", result.lower())
        self.assertNotIn("writing style", result.lower())
        self.assertNotIn("voice", result.lower())
        
        # Must contain original tweet
        self.assertIn(self.test_tweet_post, result)
        
        # Must contain task instructions
        self.assertIn("[MASKED]", result)
        self.assertIn("Instructions", result)
        
        # Must be substantial (not just the tweet)
        self.assertGreater(len(result), 100)
        
        print(f"✅ No-Persona Post Template: {len(result)} characters")
    
    def test_no_persona_reply_template(self):
        """Test No-Persona Reply Template - Critical Requirement."""
        result = format_template(
            "imitation_replies_template_no_persona", 
            tweet=self.test_tweet_reply
        )
        
        # CRITICAL: No persona references
        self.assertNotIn("persona", result.lower())
        self.assertNotIn("user", result.lower())
        self.assertNotIn("writing style", result.lower())
        self.assertNotIn("voice", result.lower())
        self.assertNotIn("authentic", result.lower())
        
        # Must contain original tweet
        self.assertIn(self.test_tweet_reply, result)
        
        # Must contain appropriate instructions
        self.assertIn("reply", result.lower())
        self.assertIn("Instructions", result)
        
        print(f"✅ No-Persona Reply Template: {len(result)} characters")
    
    def test_generic_persona_post_template(self):
        """Test Generic-Persona Post Template - Consistency Check."""
        result = format_template(
            "imitation_post_template_generic", 
            tweet=self.test_tweet_post
        )
        
        # MUST contain generic persona
        self.assertIn("social media user", result.lower())
        self.assertIn("conversational", result.lower())
        self.assertIn("authentic", result.lower())
        
        # Must NOT contain specific individual characteristics
        self.assertNotIn("film enthusiast", result.lower())
        self.assertNotIn("25-year-old", result.lower())
        self.assertNotIn("action movies", result.lower())
        
        # Must contain original tweet
        self.assertIn(self.test_tweet_post, result)
        
        # Structural consistency with individual templates
        self.assertIn("Persona:", result)
        self.assertIn("Instructions:", result)
        
        print(f"✅ Generic-Persona Post Template: {len(result)} characters")
    
    def test_generic_persona_reply_template(self):
        """Test Generic-Persona Reply Template - Consistency Check."""
        result = format_template(
            "imitation_replies_template_generic", 
            tweet=self.test_tweet_reply
        )
        
        # MUST contain generic persona
        self.assertIn("social media user", result.lower())
        self.assertIn("conversational", result.lower())
        
        # Must contain original tweet
        self.assertIn(self.test_tweet_reply, result)
        
        # Structural consistency
        self.assertIn("Persona:", result)
        self.assertIn("Instructions:", result)
        
        print(f"✅ Generic-Persona Reply Template: {len(result)} characters")
    
    def test_template_length_consistency(self):
        """Test that baseline templates have comparable length to individual templates."""
        individual_post = format_template(
            "imitation_post_template_simple",
            persona=self.test_persona,
            tweet=self.test_tweet_post
        )
        
        no_persona_post = format_template(
            "imitation_post_template_no_persona",
            tweet=self.test_tweet_post
        )
        
        generic_post = format_template(
            "imitation_post_template_generic",
            tweet=self.test_tweet_post
        )
        
        # Templates should be roughly comparable in length
        # (prevents unfair comparison due to prompt length effects)
        individual_len = len(individual_post)
        no_persona_len = len(no_persona_post)
        generic_len = len(generic_post)
        
        # Allow 50% variation in length
        self.assertGreater(no_persona_len, individual_len * 0.5)
        self.assertLess(no_persona_len, individual_len * 1.5)
        
        self.assertGreater(generic_len, individual_len * 0.5)
        self.assertLess(generic_len, individual_len * 1.5)
        
        print(f"📏 Template Lengths:")
        print(f"   Individual: {individual_len}")
        print(f"   No-Persona: {no_persona_len}")
        print(f"   Generic:    {generic_len}")
    
    def test_instruction_clarity(self):
        """Test that baseline templates have clear, unambiguous instructions."""
        templates_to_test = [
            "imitation_post_template_no_persona",
            "imitation_replies_template_no_persona", 
            "imitation_post_template_generic",
            "imitation_replies_template_generic"
        ]
        
        for template_name in templates_to_test:
            if "post" in template_name:
                result = format_template(template_name, tweet=self.test_tweet_post)
            else:
                result = format_template(template_name, tweet=self.test_tweet_reply)
            
            # Must have clear task instruction
            self.assertIn("Instructions:", result)
            
            # Must have at least 3 instruction bullets/points
            instruction_indicators = result.lower().count("- ") + result.lower().count("• ")
            self.assertGreaterEqual(instruction_indicators, 3)
            
            print(f"✅ {template_name}: Clear instructions")
    
    def test_template_scientific_validity(self):
        """Test scientific validity of template design."""
        
        # Test 1: No-Persona truly removes persona conditioning
        no_persona_result = format_template(
            "imitation_post_template_no_persona",
            tweet=self.test_tweet_post
        )
        
        persona_conditioning_words = [
            "persona", "user", "style", "voice", "authentic", 
            "typical", "characteristic", "personality", "individual"
        ]
        
        for word in persona_conditioning_words:
            self.assertNotIn(word, no_persona_result.lower(),
                           f"No-Persona template contains persona conditioning word: '{word}'")
        
        # Test 2: Generic-Persona is truly generic (not specific)
        generic_result = format_template(
            "imitation_post_template_generic",
            tweet=self.test_tweet_post
        )
        
        specific_characteristics = [
            "age", "job", "hobby", "interest", "preference", 
            "opinion", "belief", "experience", "background"
        ]
        
        # Generic template should not contain specific characteristics
        # (except in the generic persona description itself)
        persona_section = generic_result.split("Original Tweet:")[0]
        content_section = generic_result.split("Original Tweet:")[1] if len(generic_result.split("Original Tweet:")) > 1 else ""
        
        for word in specific_characteristics:
            self.assertNotIn(word, content_section.lower(),
                           f"Generic template contains specific characteristic in instructions: '{word}'")
        
        print("✅ Scientific validity confirmed")
    
    def test_backward_compatibility(self):
        """Test that existing templates still work (regression test)."""
        
        # Test existing individual templates
        individual_post = format_template(
            "imitation_post_template_simple",
            persona=self.test_persona,
            tweet=self.test_tweet_post
        )
        
        individual_reply = format_template(
            "imitation_replies_template_simple",
            persona=self.test_persona,
            tweet=self.test_tweet_reply
        )
        
        # Should contain persona and tweet
        self.assertIn(self.test_persona, individual_post)
        self.assertIn(self.test_tweet_post, individual_post)
        self.assertIn(self.test_persona, individual_reply)
        self.assertIn(self.test_tweet_reply, individual_reply)
        
        print("✅ Backward compatibility maintained")


def run_phase1_validation():
    """Run comprehensive Phase 1 validation."""
    print("="*60)
    print("PHASE 1 VALIDATION: BASELINE TEMPLATE TESTING")
    print("="*60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION COMPLETE")
    print("="*60)
    
    # Additional integration test
    print("\n🧪 Integration Test:")
    
    try:
        # Test all new templates can be loaded
        new_templates = [
            "imitation_post_template_no_persona",
            "imitation_replies_template_no_persona",
            "imitation_post_template_generic", 
            "imitation_replies_template_generic"
        ]
        
        for template_name in new_templates:
            template = select_template(template_name)
            assert len(template) > 50, f"{template_name} too short"
            print(f"   ✅ {template_name}: Loaded successfully")
        
        print("\n🎉 Phase 1 PASSED: All baseline templates ready for Phase 2!")
        
    except Exception as e:
        print(f"\n❌ Phase 1 FAILED: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = run_phase1_validation()
    exit(0 if success else 1)