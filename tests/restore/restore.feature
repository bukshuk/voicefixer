Feature: VoiceFixer Restore
  As an audio engineer,
  I want to restore antique audio recordings,
  By using VoiceFixer neural network.

  @short
  Scenario: VoiceFixer File Restore (s)
    Given the VoiceFixer model
    When I restore recording with the <index>
    Then I get the restored recording

    Examples:
    | index |
    | 12    |
    | 19    |

  @long
  Scenario: VoiceFixer File Restore (l)
    Given the VoiceFixer model
    When I restore recording with the <index>
    Then I get the restored recording

    Examples:
    | index |
    | 20    |
    | 68    |
