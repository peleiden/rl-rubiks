import { Injectable } from '@angular/core';
import { cube, cube20, IScrambleRequest, ISolveResponse, ISolveRequest } from './rubiks/rubiks';
import { HttpService } from './http.service';

function Lock() {
  // Sets CommonService.locked to true while method is applied
  // If it already is true, the method is not applied
  return function(target: CommonService, key: string | symbol, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    descriptor.value = async function(...args: any[]) {
      if (target.locked) {
        return;
      } else {
        target.locked = true;
        const result = await original.apply(this, args);
        target.locked = false;
        return result;
      }
    };
    return descriptor;
  };
}

@Injectable({
  providedIn: 'root'
})
export class CommonService {

  locked = false;
  cuda: boolean;
  agents: string[];
  timeLimit: number;
  selectedAgent: number;
  hasSearchedForSolution = false;
  hasSolution: boolean;
  solveActions: string[];

  state: cube;
  state20: cube20;
  actions = ["F", "f", "B", "b", "T", "t", "D", "d", "L", "l", "R", "r"];

  constructor(private httpService: HttpService) { }

  public async getInfo() {
    const { cuda, agents } = await this.httpService.getInfo();
    this.cuda = cuda;
    this.agents = agents;
    this.selectedAgent = 0;
    this.timeLimit = 1;
  }

  @Lock()
  public async getSolved() {
    const { state, state20 } = await this.httpService.getSolved();
    this.state = state;
    this.state20 = state20;
  }

  @Lock()
  public async act(action: number) {
    const { state, state20 } = await this.httpService.performAction({action, state20: this.state20});
    this.state = state;
    this.state20 = state20;
  }

  @Lock()
  public async scramble(depth: number) {
    const scrambleRequest: IScrambleRequest = { depth, state20: this.state20, };
    const { states, finalState20 } = await this.httpService.scramble(scrambleRequest);
    this.state20 = finalState20;
    this.animateStates(states);
  }

  @Lock()
  public async solve() {
    const solveRequest: ISolveRequest = {
      agentIdx: this.selectedAgent,
      timeLimit: this.timeLimit,
      state20: this.state20,
    };
    const { solution, actions, states, finalState20 } = await this.httpService.solve(solveRequest);
    this.hasSearchedForSolution = true;
    this.hasSolution = solution;
    this.solveActions = actions.map(val => this.actions[2*val[0] + val[1]]);
    this.state20 = finalState20;
    if (this.hasSolution) {
      this.animateStates(states);
    }
  }

  private async animateStates(states: cube[]) {
    const frameLength = Math.min(1000 / states.length, 100);
    for (const state of states) {
      const promise = new Promise<cube>((resolve, reject) => {
        setTimeout(() => resolve(state), frameLength);
      })
      this.state = await promise;
    }
  }
}
