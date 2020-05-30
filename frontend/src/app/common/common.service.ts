import { Injectable } from '@angular/core';
import { cube69, cube, ISolveRequest, action } from './rubiks';
import { HttpService } from './http.service';
import { deepCopy, CubeService } from './cube.service';

@Injectable({
  providedIn: 'root'
})
export class CommonService {

  status = {
    loading: true,
    hasData: false,
  };
  cuda: boolean;
  searchers: string[];
  timeLimit = 1;
  selectedSearcher: number;
  hasSearchedForSolution = false;
  hasSolution: boolean;
  exploredStates: number;
  actionQueue: number[];
  solveLength: number;

  constructor(private httpService: HttpService, private cubeService: CubeService) { }

  get prettyActionQueue(): string {
    return this.actionQueue.map(val => this.cubeService.actions[val]).join(" ");
  }

  public setSelectedHost(host: { name: string, address: string }) {
    this.httpService.selectedHost = host;
    this.getInfo();
  }

  public scramble(moves: number) {
    // Scrambles from current state with animation
    const actions = this.cubeService.scramble(moves);
    this.animateStates(actions);
  }

  public async getInfo() {
    try {
      this.status.loading = true;
      const { cuda, searchers } = await this.httpService.getInfo();
      this.cuda = cuda;
      this.searchers = searchers;
      this.selectedSearcher = 0;
      this.status.hasData = true;
      this.status.loading = false;
    } catch (e) {
      this.status.hasData = false;
      this.status.loading = false;
    }
  }

  public async solve() {
    if (this.status.loading) return;
    this.status.loading = true;
    const solveRequest: ISolveRequest = {
      agentIdx: this.selectedSearcher,
      timeLimit: this.timeLimit,
      state: this.cubeService.currentState,
    };
    const { solution, actions, exploredStates } = await this.httpService.solve(solveRequest);
    this.hasSearchedForSolution = true;
    this.hasSolution = solution;
    this.exploredStates = exploredStates;
    this.actionQueue = actions;
    this.actionQueue.reverse();  // Allows pop to be used instead of shift which changes runtime complexity from O(n**2) to O(n)
    this.solveLength = this.actionQueue.length;
    this.status.loading = false;
  }

  public getStates(actions: number[]): cube[] {
    const states: cube[] = [];
    let state = this.cubeService.currentState;
    for (let action of actions) {
      state = this.cubeService.rotate(state, ...this.cubeService.getAction(action));
      states.push(state);
    }
    return states;
  }

  public async animateStates(actions: number[] = null) {
    // Actions should go last to first
    // Uses this.actionQueue if actions is null
    this.status.loading = true;
    const remainingActions = deepCopy(actions) || this.actionQueue;
    const frameLength = Math.min(1000 / remainingActions.length, 100);
    while (remainingActions.length) {
      const promise = new Promise((resolve, reject) => { setTimeout(() => resolve(), frameLength); })
      await promise;
      const action = remainingActions.pop();
      this.step(action);
    }
    this.status.loading = false;
  }

  public step(action: number = null) {
    if (action === null) {
      action = this.actionQueue.pop();
    }
    this.cubeService.currentState = this.cubeService.rotate(this.cubeService.currentState, ...this.cubeService.getAction(action));
  }
}
