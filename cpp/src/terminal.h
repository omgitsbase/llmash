#pragma once

namespace llmash {

// Reports an arrow as 0xE0 then a code on every platform, the way _getch does.
int raw_key();

int terminal_columns();

bool stdin_is_terminal();
bool stdout_is_terminal();
bool stderr_is_terminal();

} // namespace llmash
