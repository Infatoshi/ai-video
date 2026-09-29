// The concept words: when one is sung, the overlay slams it on screen as the biggest thing in the frame
// (hud.ts). Keyed by lyric line (a fragment that finds it) and the sung word it lands on; `show` is what
// prints, so the lyric can say "four thousand" while the page says the exact 4,096.
export interface Slam { line: string; word: string; show: string; nth?: number }

export const SLAMS: Slam[] = [
  // verse 1: text -> tokens -> vectors
  { line: 'Type a question', word: 'enter', show: 'ENTER' },
  { line: 'Chopped up into tokens', word: 'tokens', show: 'TOKENS' },
  { line: "It don't see letters", word: 'chunks', show: 'CHUNKS' },
  { line: 'count the R', word: 'strawberry', show: '1 TOKEN' },   // " strawberry" is one token in the question
  { line: 'Every token is a vector', word: 'vector', show: 'VECTOR' },
  { line: 'Every token is a vector', word: 'thousand', show: '4,096' },
  { line: 'Words that mean the same', word: 'side', show: 'SIDE BY SIDE' },
  { line: 'stack of numbers', word: 'numbers', show: 'NUMBERS' },
  { line: 'Thirty-two floors', word: 'thirty', show: '32 FLOORS' },
  // chorus (x3)
  ...[0, 1, 2].flatMap((nth) => [
    { line: 'Just one more token', word: 'token', show: 'ONE MORE TOKEN', nth },
    { line: 'One more roll of the dice', word: 'dice', show: 'DICE', nth },
    { line: 'Guess it, pick it', word: 'pick', show: 'PICK', nth },
    { line: 'Guess it, pick it', word: 'feed', show: 'FEED IT BACK', nth },
    { line: 'do it again all night', word: 'again', show: 'AGAIN', nth },
  ]),
  // verse 2: attention, feed-forward, the stack
  { line: 'looks back at the words', word: 'back', show: 'LOOK BACK' },
  { line: 'who matters to me', word: 'weight', show: 'WEIGHT' },
  { line: 'The strongest ones pull', word: 'blends', show: 'BLEND' },
  { line: "That's attention", word: 'attention', show: 'ATTENTION' },
  { line: "That's attention", word: 'heads', show: '32 HEADS' },
  { line: 'feed-forward fires', word: 'facts', show: 'FACTS' },
  { line: 'Fourteen thousand neurons', word: 'neurons', show: '14,336' },
  { line: 'one floor of the tower', word: 'floor', show: '1 FLOOR' },
  { line: 'Stack it thirty-two high', word: 'power', show: '×32' },
  // verse 3: logits -> sampling -> the KV cache
  { line: 'every word gets a score', word: 'score', show: 'SCORE' },
  { line: 'A hundred twenty-eight thousand', word: 'thousand', show: '128,256' },
  { line: 'Softmax turns them', word: 'softmax', show: 'SOFTMAX' },
  { line: 'Softmax turns them', word: 'temperature', show: 'TEMPERATURE' },
  { line: 'Keep it low', word: 'safe', show: 'SAFE' },
  { line: 'Keep it low', word: 'bet', show: 'BET' },
  { line: 'Roll the dice', word: 'end', show: 'APPEND' },
  { line: 'Feed the whole thing back', word: 'again', show: 'AGAIN' },
  { line: 'keys and the values', word: 'keep', show: 'KEEP' },
  { line: 'That\'s the K', word: 'cache', show: 'KV CACHE' },
  { line: 'That\'s the K', word: 'cheap', show: 'CHEAP' },
  // breakdown: memory-bound decode
  { line: 'Every single token, it reads', word: 'weight', show: 'EVERY WEIGHT' },
  { line: 'Sixteen gigs', word: 'sixteen', show: '16.06 GB' },   // the weights; a decode step reads 15.01 GB of them (memory.ts)
  { line: 'Sixteen gigs', word: 'waits', show: 'WAIT' },
  // verse 4: batching, speculative decoding
  { line: 'Stuck on the slope', word: 'roofline', show: 'ROOFLINE' },
  { line: 'read the weights once', word: 'hundred', show: '×100' },
  { line: 'a little model guesses', word: 'guesses', show: 'GUESS' },
  { line: 'checks all four', word: 'go', show: 'ONE PASS' },
  // outro
  { line: 'says it', word: 'done', show: 'DONE' },
];
